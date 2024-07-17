use std::path::{Path, PathBuf};

use anyhow::Result;
use image::ImageFormat;
use indicatif::{ProgressIterator, ProgressStyle};
use walkdir::WalkDir;

use crate::image_processor::ImageProcessor;
use crate::model::Predictor;

pub struct ProgressTracker {
    image_paths: Vec<Vec<PathBuf>>,
}

impl ProgressTracker {
    pub fn new<P: AsRef<Path>>(input_dir: P, batch_size: usize) -> Self {
        let image_paths: Vec<_> = WalkDir::new(input_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| ImageFormat::from_path(e.path()).is_ok())
            .map(|e| e.into_path())
            .collect();
        let batched_paths: Vec<_> = image_paths.chunks(batch_size).map(|c| c.to_vec()).collect();

        Self {
            image_paths: batched_paths,
        }
    }

    pub fn process(self, processor: &ImageProcessor, model: &Predictor) -> Result<()> {
        let style = ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-");

        self.image_paths
            .iter()
            .progress_with_style(style)
            .map(|batch| processor.process(batch, model))
            .collect::<Result<Vec<_>>>()?;

        Ok(())
    }
}
