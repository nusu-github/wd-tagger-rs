use std::path::Path;

use anyhow::{ensure, Result};
use clap::Parser;

use crate::image_processor::ImageProcessor;
use crate::labels::LabelAnalyzer;
use crate::model::Predictor;
use crate::progress_tracker::ProgressTracker;

mod image_processor;
mod labels;
mod model;
mod progress_tracker;

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Config {
    #[arg(short, long)]
    input_dir: String,

    #[arg(short, long)]
    output_dir: String,

    #[arg(short, long)]
    model_path: String,

    #[arg(short, long)]
    csv_path: String,

    #[arg(short, long, default_value_t = 0)]
    device_id: i32,

    #[arg(short, long, default_value_t = 1)]
    batch_size: usize,

    #[arg(long, default_value_t = 0.35)]
    general_threshold: f32,

    #[arg(long, action)]
    general_mcut_enabled: bool,

    #[arg(long, default_value_t = 0.85)]
    character_threshold: f32,

    #[arg(long, action)]
    character_mcut_enabled: bool,
}

fn main() -> Result<()> {
    let config = Config::parse();
    ensure!(
        Path::new(&config.model_path).exists(),
        "Model path does not exist"
    );
    ensure!(
        Path::new(&config.input_dir).exists(),
        "Input directory does not exist"
    );

    let label_analyzer = LabelAnalyzer::new(&config.csv_path)?;
    let model = Predictor::new(&config.model_path, config.device_id, label_analyzer)?;
    let image_processor = ImageProcessor::new(&config);
    let progress_tracker = ProgressTracker::new(&config.input_dir, config.batch_size);
    progress_tracker.process(image_processor, model);
    Ok(())
}
