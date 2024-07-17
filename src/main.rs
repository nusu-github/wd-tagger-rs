use std::path::PathBuf;

use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;

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
    #[arg()]
    input_dir: PathBuf,

    #[arg()]
    output_dir: PathBuf,

    #[arg(short, long, group = "model", value_enum, default_value_t = ModelName::WdSwinv2TaggerV3)]
    model_name: ModelName,

    #[arg(long, group = "model")]
    model_path: Option<PathBuf>,

    #[arg(long, group = "model")]
    csv_path: Option<PathBuf>,

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

#[derive(Debug, Clone, ValueEnum)]
enum ModelName {
    WdSwinv2TaggerV3 = 0,
    WdConvnextTaggerV3 = 1,
    WdVitTaggerV3 = 2,
    WdV14MoatTaggerV2 = 3,
    WdV14Convnextv2TaggerV2 = 4,
    WdV14Swinv2TaggerV2 = 5,
    WdV14ConvnextTaggerV2 = 6,
    WdV14VitTaggerV2 = 7,
    WdV14ConvnextTagger = 8,
    WdV14VitTagger = 9,
}

pub const MODELS: &[&str] = &[
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
    "SmilingWolf/wd-vit-tagger-v3",
    "SmilingWolf/wd-v1-4-moat-tagger-v2",
    "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "SmilingWolf/wd-v1-4-vit-tagger-v2",
    "SmilingWolf/wd-v1-4-convnext-tagger",
    "SmilingWolf/wd-v1-4-vit-tagger",
];

fn main() -> Result<()> {
    let config = Config::parse();
    ensure!(&config.input_dir.exists(), "Input directory does not exist");

    let (model_path, csv_path) = match &config.model_path {
        None => {
            let api = Api::new()?;
            let repo = api.model(MODELS[(&config.model_name).to_owned() as usize].to_string());
            let model_path = repo.get("model.onnx")?;
            let csv_path = repo.get("selected_tags.csv")?;

            (model_path, csv_path)
        }
        Some(model_path) => {
            let csv_path = match &config.csv_path {
                None => {
                    eprintln!("CSV path is required when model path is provided");
                    std::process::exit(1);
                }
                Some(csv_path) => csv_path,
            };

            (model_path.to_owned(), csv_path.to_owned())
        }
    };

    let label_analyzer = LabelAnalyzer::new(&csv_path)?;
    let model = Predictor::new(&model_path, config.device_id, label_analyzer)?;
    let image_processor = ImageProcessor::new(&config);
    let progress_tracker = ProgressTracker::new(&config.input_dir, config.batch_size);
    progress_tracker.process(&image_processor, &model)?;

    Ok(())
}
