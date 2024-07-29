use std::path::Path;

use anyhow::Result;
use image::{imageops, Rgb, RgbImage};
use ndarray::prelude::*;
use ort::Session;

pub struct Model {
    session: Session,
    input_name: &'static str,
    output_name: &'static str,
    pub target_size: u32,
}

impl Model {
    pub fn new<P: AsRef<Path>>(model_path: P, device_id: i32) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers([ort::CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?
            .commit_from_file(model_path)?;

        let target_size = session.inputs[0].input_type.tensor_dimensions().unwrap()[1] as u32;
        let input_name = session.inputs[0].name.clone();
        let output_name = session.outputs[0].name.clone();

        Ok(Self {
            session,
            input_name: Box::leak(input_name.into_boxed_str()) as &str,
            output_name: Box::leak(output_name.into_boxed_str()) as &str,
            target_size,
        })
    }

    pub fn predict(&self, inputs: ArrayView4<f32>) -> Result<Array2<f32>> {
        let outputs = self.session.run(ort::inputs![self.input_name => inputs]?)?;
        Ok(outputs[self.output_name]
            .try_extract_tensor::<f32>()?
            .into_dimensionality::<Ix2>()?
            .to_owned())
    }
}

pub fn preprocess(image: &RgbImage, target_size: u32) -> Result<Array3<f32>> {
    let (w, h) = image.dimensions();
    let max_dim = w.max(h);
    let pad = |x| ((max_dim - x) / 2) as i64;

    let mut padded = RgbImage::from_pixel(max_dim, max_dim, Rgb([255, 255, 255]));
    imageops::overlay(&mut padded, image, pad(w), pad(h));

    let resized = imageops::resize(
        &padded,
        target_size,
        target_size,
        imageops::FilterType::Lanczos3,
    );

    let tensor = Array3::from_shape_vec(
        (target_size as usize, target_size as usize, 3),
        resized.into_raw(),
    )?
    .mapv(|x| x as f32);

    Ok(tensor)
}
