use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use crossbeam_channel::{bounded, unbounded};
use hf_hub::api::sync::Api;
use image::ImageFormat;
use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};
use ndarray::{stack, Axis};
use walkdir::WalkDir;

use crate::model::preprocess;

mod labels;
mod model;

#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
pub struct Config {
    input_dir: PathBuf,
    output_dir: PathBuf,
    #[arg(short, long, group = "model", value_enum, default_value_t = ModelName::WdSwinv2TaggerV3)]
    model_name: ModelName,
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

#[derive(ValueEnum, Clone)]
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

const MODELS: &[&str] = &[
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

    let api = Api::new()?;
    let repo = api.model(MODELS[(&config.model_name).to_owned() as usize].to_string());
    let model_path = repo.get("model.onnx")?;
    let csv_path = repo.get("selected_tags.csv")?;

    let label_analyzer = labels::LabelAnalyzer::new(
        &csv_path,
        config.general_threshold,
        config.general_mcut_enabled,
        config.character_threshold,
        config.character_mcut_enabled,
    )?;
    let model = model::Model::new(&model_path, config.device_id)?;
    let target_size = model.target_size;

    let image_paths: Vec<_> = WalkDir::new(config.input_dir.as_path())
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| ImageFormat::from_path(e.path()).is_ok())
        .map(|e| e.into_path())
        .collect();
    let batched_paths: Vec<_> = image_paths.chunks(config.batch_size).collect();

    let (load_bar, inference_bar, output_bar) =
        progress(image_paths.len() as u64, batched_paths.len() as u64)?;

    let (load_tx, load_rx) = bounded(config.batch_size);
    let (inference_tx, inference_rx) = unbounded();

    rayon::scope(|s| {
        s.spawn(move |s| {
            for paths in batched_paths {
                let (result_tx, result_rx) = unbounded();
                for (i, path) in paths.iter().enumerate() {
                    s.spawn({
                        let result_tx = result_tx.clone();
                        let load_bar = load_bar.clone();
                        move |_| {
                            load_bar.set_message(path.display().to_string());
                            let image = image::open(path).unwrap().into_rgb8();
                            let tensor = preprocess(&image, target_size).unwrap();
                            result_tx.send((i, tensor)).unwrap();

                            load_bar.inc(1);
                        }
                    })
                }
                drop(result_tx);

                let mut batch: Vec<_> = result_rx.iter().collect();
                batch.sort_by(|a, b| a.0.cmp(&b.0));
                let batch = batch.into_iter().map(|(_, x)| x).collect::<Vec<_>>();

                let batch =
                    stack(Axis(0), &batch.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
                load_tx.send((batch, paths)).unwrap();
            }
        });

        s.spawn(move |_| {
            while let Ok((batch, paths)) = load_rx.recv() {
                let labels = model.predict(batch.view()).unwrap();
                let labels: Vec<_> = labels
                    .axis_iter(Axis(0))
                    .map(|x| label_analyzer.analyze(x.as_slice().unwrap()))
                    .collect();
                inference_tx.send((labels, paths)).unwrap();

                inference_bar.inc(1);
            }
        });

        s.spawn(move |_| {
            let input_dir = config.input_dir.as_path();
            let output_dir = config.output_dir.as_path();

            while let Ok((labels, paths)) = inference_rx.recv() {
                let output_dirs: Vec<_> = paths
                    .iter()
                    .map(|p| {
                        let relative = p.strip_prefix(input_dir).unwrap();
                        let output_subdir =
                            output_dir.join(relative.parent().unwrap_or(Path::new("")));
                        fs::create_dir_all(&output_subdir).unwrap();
                        output_subdir
                    })
                    .collect();

                for (i, (_ratings, general, character)) in labels.iter().enumerate() {
                    let filename = paths[i].file_stem().unwrap().to_str().unwrap();
                    let output_file = output_dirs[i].join(format!("{filename}.txt"));
                    let tags: Vec<_> = general
                        .iter()
                        .map(|(t, _)| t.to_string())
                        .chain(character.iter().map(|(t, _)| t.to_string()))
                        .collect();
                    fs::write(output_file, tags.join(", ")).unwrap();

                    output_bar.inc(1);
                }
            }
        });
    });

    Ok(())
}

fn progress(
    image_paths_count: u64,
    batched_paths_count: u64,
) -> Result<(ProgressBar, ProgressBar, ProgressBar)> {
    let mp = MultiProgress::new();

    let load_bar = mp
        .add(ProgressBar::new(image_paths_count))
        .with_finish(ProgressFinish::AbandonWithMessage("Completed!".into()));
    load_bar.set_style(ProgressStyle::default_bar()
        .template("Preprocessing: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
        .progress_chars("#>-"));

    let inference_bar = mp
        .add(ProgressBar::new(batched_paths_count))
        .with_finish(ProgressFinish::AbandonWithMessage("Completed!".into()));
    inference_bar.set_style(ProgressStyle::default_bar()
        .template("Inference: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));

    let output_bar = mp
        .add(ProgressBar::new(image_paths_count))
        .with_finish(ProgressFinish::AbandonWithMessage("Completed!".into()));
    output_bar.set_style(ProgressStyle::default_bar()
        .template("Output: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
        .progress_chars("#>-"));

    mp.set_move_cursor(true);
    Ok((load_bar, inference_bar, output_bar))
}
