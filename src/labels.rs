use std::path::Path;

use anyhow::Result;
use ndarray::prelude::*;

#[derive(serde::Deserialize)]
struct Label {
    name: String,
    category: u32,
}

pub struct LabelAnalyzer {
    tags: Vec<String>,
    rating_indices: Vec<usize>,
    general_indices: Vec<usize>,
    character_indices: Vec<usize>,
}

pub type TagScores<'a> = Vec<(&'a str, f32)>;

fn mcut_threshold(probs: &Vec<&f32>) -> f32 {
    let mut sorted_probs = probs.to_owned();
    sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let t = sorted_probs
        .windows(2)
        .map(|w| w[0] - w[1])
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    (sorted_probs[t] + sorted_probs[t + 1]) / 2.0
}

impl LabelAnalyzer {
    pub fn new<P>(csv_path: &P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        const KAOMOJIS: &[&str] = &[
            "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3",
            "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
        ];
        const RATING: u32 = 9;
        const GENERAL: u32 = 0;
        const CHARACTER: u32 = 4;

        let mut reader = csv::Reader::from_path(csv_path)?;
        let mut tags = Vec::new();
        let mut rating_indices = Vec::new();
        let mut general_indices = Vec::new();
        let mut character_indices = Vec::new();
        for (i, result) in reader.deserialize().enumerate() {
            let label: Label = result?;
            let tag = match KAOMOJIS.contains(&label.name.as_str()) {
                true => label.name,
                false => label.name.replace('_', ""),
            };
            tags.push(tag);
            match label.category {
                RATING => rating_indices.push(i),
                GENERAL => general_indices.push(i),
                CHARACTER => character_indices.push(i),
                _ => (),
            }
        }
        Ok(Self {
            tags,
            rating_indices,
            general_indices,
            character_indices,
        })
    }

    pub fn analyze(
        &self,
        preds: ArrayView1<f32>,
        general_threshold: f32,
        general_mcut_enabled: bool,
        character_threshold: f32,
        character_mcut_enabled: bool,
    ) -> (TagScores, TagScores, TagScores) {
        let mut ratings: Vec<_> = self
            .rating_indices
            .iter()
            .map(|&i| (self.tags[i].as_str(), preds[i]))
            .collect();
        ratings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let general_threshold = if general_mcut_enabled {
            0.0
        } else {
            general_threshold
        };
        let character_threshold = if character_mcut_enabled {
            0.0
        } else {
            character_threshold
        };

        let mut general: Vec<_> = self
            .general_indices
            .iter()
            .filter(|&&i| preds[i] >= general_threshold)
            .map(|&i| (self.tags[i].as_str(), preds[i]))
            .collect();

        if general_mcut_enabled {
            let general_probs: Vec<_> = general.iter().map(|(_, p)| p).collect();
            let general_threshold = mcut_threshold(&general_probs);
            general = general
                .into_iter()
                .filter(|&(_, p)| p >= general_threshold)
                .collect();
        }
        general.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut character: Vec<_> = self
            .character_indices
            .iter()
            .filter(|&&i| preds[i] >= character_threshold)
            .map(|&i| (self.tags[i].as_str(), preds[i]))
            .collect();

        if character_mcut_enabled {
            let character_probs: Vec<_> = character.iter().map(|(_, p)| p).collect();
            let character_threshold = mcut_threshold(&character_probs);
            character = character
                .into_iter()
                .filter(|&(_, p)| p >= character_threshold)
                .collect();
        }
        character.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        (ratings, general, character)
    }
}
