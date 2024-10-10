use std::path::Path;

use anyhow::Result;

const KAOMOJIS: &[&str] = &[
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o",
    "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
];
const RATING: u32 = 9;
const GENERAL: u32 = 0;
const CHARACTER: u32 = 4;

#[derive(serde::Deserialize)]
struct Label {
    name: String,
    category: u32,
}

pub struct LabelAnalyzer {
    tags: Vec<&'static str>,
    rating_indices: Vec<usize>,
    general_indices: Vec<usize>,
    character_indices: Vec<usize>,
    general_threshold: f32,
    general_mcut_enabled: bool,
    character_threshold: f32,
    character_mcut_enabled: bool,
}

pub type TagScores = Vec<(&'static str, f32)>;

impl LabelAnalyzer {
    pub fn new<P: AsRef<Path>>(
        csv_path: P,
        general_threshold: f32,
        general_mcut_enabled: bool,
        character_threshold: f32,
        character_mcut_enabled: bool,
    ) -> Result<Self> {
        let mut reader = csv::Reader::from_path(csv_path)?;
        let len = reader.headers()?.len() - 1;
        let mut tags: Vec<&'static str> = Vec::with_capacity(len);
        let mut rating_indices = Vec::with_capacity(len);
        let mut general_indices = Vec::with_capacity(len);
        let mut character_indices = Vec::with_capacity(len);

        for (i, result) in reader.deserialize().enumerate() {
            let label: Label = result?;
            let tag = if KAOMOJIS.contains(&label.name.as_str()) {
                label.name
            } else {
                label.name.replace('_', "")
            };
            tags.push(Box::leak(tag.into_boxed_str()));
            match label.category {
                RATING => rating_indices.push(i),
                GENERAL => general_indices.push(i),
                CHARACTER => character_indices.push(i),
                _ => unreachable!(),
            }
        }

        Ok(Self {
            tags,
            rating_indices,
            general_indices,
            character_indices,
            general_threshold,
            general_mcut_enabled,
            character_threshold,
            character_mcut_enabled,
        })
    }

    pub fn analyze(&self, preds: &[f32]) -> (TagScores, TagScores, TagScores) {
        let mut ratings = self.tag_scores(preds, &self.rating_indices);
        ratings.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let general = self.tag_scores(preds, &self.general_indices);
        let mut general = filter_tags(general, self.general_threshold, self.general_mcut_enabled);
        general.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let character = self.tag_scores(preds, &self.character_indices);
        let mut character = filter_tags(
            character,
            self.character_threshold,
            self.character_mcut_enabled,
        );
        character.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        (ratings, general, character)
    }

    fn tag_scores(&self, preds: &[f32], indices: &[usize]) -> TagScores {
        indices.iter().map(|&i| (self.tags[i], preds[i])).collect()
    }
}

fn filter_tags(tag_scores: TagScores, threshold: f32, mcut_enabled: bool) -> TagScores {
    let threshold = if mcut_enabled {
        let scores_probs: Vec<_> = tag_scores.iter().map(|(_, p)| *p).collect();
        mcut_threshold(&scores_probs)
    } else {
        threshold
    };

    tag_scores
        .into_iter()
        .filter(|(_, p)| *p >= threshold)
        .collect()
}

fn mcut_threshold(prediction: &[f32]) -> f32 {
    let mut sorted = prediction.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let difs: Vec<_> = sorted.windows(2).map(|w| w[0] - w[1]).collect();

    let t = difs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map_or(0, |(i, _)| i);

    (sorted[t] + sorted[t + 1]) / 2.0
}
