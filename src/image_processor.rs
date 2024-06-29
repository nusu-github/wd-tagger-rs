use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::Config;
use crate::model::Predictor;

pub struct ImageProcessor {
    input_dir: PathBuf,
    output_dir: PathBuf,
    general_threshold: f32,
    general_mcut_enabled: bool,
    character_threshold: f32,
    character_mcut_enabled: bool,
}

impl ImageProcessor {
    pub fn new(config: &Config) -> Self {
        Self {
            input_dir: Path::new(&config.input_dir).to_path_buf(),
            output_dir: Path::new(&config.output_dir).to_path_buf(),
            general_threshold: config.general_threshold,
            general_mcut_enabled: config.general_mcut_enabled,
            character_threshold: config.character_threshold,
            character_mcut_enabled: config.character_mcut_enabled,
        }
    }

    pub fn process(&self, paths: Vec<PathBuf>, model: &Predictor) -> Result<()> {
        let output_dirs: Vec<_> = paths
            .iter()
            .map(|p| {
                let relative = p.strip_prefix(&self.input_dir).unwrap();
                let output_subdir = self
                    .output_dir
                    .join(relative.parent().unwrap_or(Path::new("")));
                fs::create_dir_all(&output_subdir).unwrap();
                output_subdir
            })
            .collect();

        let images: Vec<_> = paths
            .par_iter()
            .map(|p| image::open(p).with_context(|| p.display().to_string()))
            .collect::<Result<_>>()?;
        let batch = model.preprocess(images)?;
        let labels = model.predict(
            batch.view(),
            self.general_threshold,
            self.general_mcut_enabled,
            self.character_threshold,
            self.character_mcut_enabled,
        )?;

        for (i, (_ratings, general, character)) in labels.iter().enumerate() {
            let filename = paths[i].file_stem().unwrap().to_string_lossy();
            let output_file = output_dirs[i].join(format!("{}.txt", filename));
            let tags: Vec<_> = general
                .iter()
                .map(|(t, _)| t.to_string())
                .chain(character.iter().map(|(t, _)| t.to_string()))
                .collect();
            fs::write(output_file, tags.join(", "))?;
        }
        Ok(())
    }
}
