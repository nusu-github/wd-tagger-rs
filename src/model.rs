use std::path::Path;

use anyhow::Result;
use image::{imageops, Rgb, RgbImage};
use ndarray::prelude::*;
use nshare::AsNdarray3;
use num_traits::AsPrimitive;
use ort::Session;

use crate::padding::Padding;

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
            input_name: Box::leak(input_name.into_boxed_str()),
            output_name: Box::leak(output_name.into_boxed_str()),
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

pub fn preprocess(image: RgbImage, target_size: u32) -> Array3<f32> {
    let resized = imageops::resize(
        &image,
        target_size,
        target_size,
        imageops::FilterType::Lanczos3,
    );
    let (padded, _) = resized.padding_square(Rgb([255, 255, 255]));
    padded
        .as_ndarray3()
        .mapv(|x| x.as_())
        .permuted_axes([1, 2, 0]) // CHW -> HWC
}
