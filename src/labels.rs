use std::path::Path;
use std::sync::OnceLock;

use anyhow::Result;

const KAOMOJIS: &[&str] = &[
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o",
    "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
];
const RATING: u32 = 9;
const GENERAL: u32 = 0;
const CHARACTER: u32 = 4;

static TAGS: OnceLock<&[&str]> = OnceLock::new();
static RATING_INDICES: OnceLock<&[usize]> = OnceLock::new();
static GENERAL_INDICES: OnceLock<&[usize]> = OnceLock::new();
static CHARACTER_INDICES: OnceLock<&[usize]> = OnceLock::new();

#[derive(serde::Deserialize)]
struct Label {
    name: String,
    category: u32,
}

pub struct LabelAnalyzer {
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
        let mut tags = Vec::with_capacity(len);
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
            tags.push(Box::leak(tag.into_boxed_str()) as &str);
            match label.category {
                RATING => rating_indices.push(i),
                GENERAL => general_indices.push(i),
                CHARACTER => character_indices.push(i),
                _ => unreachable!(),
            }
        }

        TAGS.set(Box::leak(tags.into_boxed_slice())).unwrap();
        RATING_INDICES
            .set(Box::leak(rating_indices.into_boxed_slice()))
            .unwrap();
        GENERAL_INDICES
            .set(Box::leak(general_indices.into_boxed_slice()))
            .unwrap();
        CHARACTER_INDICES
            .set(Box::leak(character_indices.into_boxed_slice()))
            .unwrap();

        Ok(Self {
            general_threshold,
            general_mcut_enabled,
            character_threshold,
            character_mcut_enabled,
        })
    }

    pub fn analyze(&self, preds: &[f32]) -> (TagScores, TagScores, TagScores) {
        let mut ratings = tag_scores(preds, RATING_INDICES.get().unwrap());
        ratings.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let general = tag_scores(preds, GENERAL_INDICES.get().unwrap());
        let mut general = filter_tags(general, self.general_threshold, self.general_mcut_enabled);
        general.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let character = tag_scores(preds, CHARACTER_INDICES.get().unwrap());
        let mut character = filter_tags(
            character,
            self.character_threshold,
            self.character_mcut_enabled,
        );
        character.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        (ratings, general, character)
    }
}

fn tag_scores(preds: &[f32], indices: &[usize]) -> TagScores {
    let tags = TAGS.get().unwrap();
    indices.iter().map(|&i| (tags[i], preds[i])).collect()
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
        .map(|(i, _)| i)
        .unwrap_or(0);

    (sorted[t] + sorted[t + 1]) / 2.0
}
