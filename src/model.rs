use std::{path::Path, sync::Mutex};

use anyhow::Result;
use image::{imageops, DynamicImage, EncodableLayout, Rgb, RgbImage};
use ndarray::{concatenate, prelude::*};
use num_traits::AsPrimitive;
use ort::{CUDAExecutionProvider, Session};

use crate::labels::{LabelAnalyzer, TagScores};

struct ImagePreprocessor {
    target_size: u32,
}

impl ImagePreprocessor {
    fn new(target_size: u32) -> Self {
        Self { target_size }
    }

    fn preprocess(&self, image: RgbImage) -> Result<Array4<f32>> {
        let (w, h) = image.dimensions();
        let max_dim = w.max(h);
        let pad = |x| <_ as AsPrimitive<_>>::as_((max_dim - x) / 2);
        let mut padded = RgbImage::from_pixel(max_dim, max_dim, Rgb([255, 255, 255]));
        imageops::overlay(&mut padded, &image, pad(w), pad(h));
        let resized = match max_dim != self.target_size {
            true => imageops::resize(
                &padded,
                self.target_size,
                self.target_size,
                imageops::FilterType::Lanczos3,
            ),
            false => padded,
        };
        let tensor = ArrayView3::from_shape(
            (self.target_size.as_(), self.target_size.as_(), 3),
            resized.as_bytes(),
        )?
        .slice(s![NewAxis,.., .., ..;-1])
        .mapv(AsPrimitive::as_);

        Ok(tensor)
    }
}

pub struct Predictor {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
    preprocessor: ImagePreprocessor,
    label_analyzer: LabelAnalyzer,
}

impl Predictor {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        device_id: i32,
        label_analyzer: LabelAnalyzer,
    ) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?
            .commit_from_file(model_path)?;

        let target_size = session.inputs[0].input_type.tensor_dimensions().unwrap()[1].as_();
        let input_name = session.inputs[0].name.to_string();
        let output_name = session.outputs[0].name.to_string();
        Ok(Self {
            session: Mutex::new(session),
            input_name,
            output_name,
            preprocessor: ImagePreprocessor::new(target_size),
            label_analyzer,
        })
    }

    pub fn preprocess(&self, images: Vec<DynamicImage>) -> Result<Array4<f32>> {
        let preprocessed = images
            .into_iter()
            .map(|img| self.preprocessor.preprocess(img.into_rgb8()))
            .collect::<Result<Vec<_>>>()?;
        let batch = concatenate(
            Axis(0),
            &preprocessed.iter().map(|t| t.view()).collect::<Vec<_>>(),
        )?;
        Ok(batch)
    }

    pub fn predict(
        &self,
        batch: Array4<f32>,
        general_threshold: f32,
        general_mcut_enabled: bool,
        character_threshold: f32,
        character_mcut_enabled: bool,
    ) -> Result<Vec<(TagScores, TagScores, TagScores)>> {
        let session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![&self.input_name => batch]?)?;
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
