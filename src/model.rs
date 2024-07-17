use std::path::Path;

use anyhow::Result;
use image::{imageops, DynamicImage, Rgb, RgbImage};
use ndarray::{prelude::*, stack};
use num_traits::AsPrimitive;
use ort::Session;
use rayon::prelude::*;

use crate::labels::{LabelAnalyzer, TagScores};

pub struct Predictor {
    session: Session,
    input_name: String,
    output_name: String,
    target_size: u32,
    label_analyzer: LabelAnalyzer,
}

impl Predictor {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        device_id: i32,
        label_analyzer: LabelAnalyzer,
    ) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([ort::CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?
            .commit_from_file(model_path)?;

        let target_size = session.inputs[0].input_type.tensor_dimensions().unwrap()[1].as_();
        let input_name = session.inputs[0].name.to_string();
        let output_name = session.outputs[0].name.to_string();

        Ok(Self {
            session,
            input_name,
            output_name,
            target_size,
            label_analyzer,
        })
    }

    pub fn preprocess(&self, images: &[DynamicImage]) -> Result<Array4<f32>> {
        let preprocessed = images
            .par_iter()
            .map(|img| preprocess(img, self.target_size))
            .collect::<Result<Vec<_>>>()?;
        let batch = stack(
            Axis(0),
            &preprocessed.iter().map(|t| t.view()).collect::<Vec<_>>(),
        )?;
        Ok(batch)
    }

    pub fn predict(
        &self,
        batch: ArrayView4<f32>,
        general_threshold: f32,
        general_mcut_enabled: bool,
        character_threshold: f32,
        character_mcut_enabled: bool,
    ) -> Result<Vec<(TagScores, TagScores, TagScores)>> {
        let outputs = self.session.run(ort::inputs![&self.input_name => batch]?)?;
        let preds = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<Ix2>()?;

        Ok(preds
            .axis_iter(Axis(0))
            .map(|p| {
                self.label_analyzer.analyze(
                    p,
                    general_threshold,
                    general_mcut_enabled,
                    character_threshold,
                    character_mcut_enabled,
                )
            })
            .collect())
    }
}

fn preprocess(image: &DynamicImage, target_size: u32) -> Result<Array3<f32>> {
    let image = image.to_rgb8();
    let (w, h) = image.dimensions();
    let max_dim = w.max(h);
    let pad = |x| <_ as AsPrimitive<_>>::as_((max_dim - x) / 2);
    let mut padded = RgbImage::from_pixel(max_dim, max_dim, Rgb([255, 255, 255]));
    imageops::overlay(&mut padded, &image, pad(w), pad(h));
    let resized = match max_dim != target_size {
        true => imageops::resize(
            &padded,
            target_size,
            target_size,
            imageops::FilterType::Lanczos3,
        ),
        false => padded,
    };
    let tensor = unsafe {
        ArrayView3::from_shape_ptr((target_size.as_(), target_size.as_(), 3), resized.as_ptr())
    }
    .slice(s![ .., .., ..;-1])
    .mapv(AsPrimitive::as_);

    Ok(tensor)
}
