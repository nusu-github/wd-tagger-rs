use image::{imageops, ImageBuffer, Pixel, Primitive};
use num_traits::AsPrimitive;

#[allow(dead_code)]
pub enum Position {
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

pub fn to_position(size: [u32; 2], pad_size: [u32; 2], position: &Position) -> Option<(i64, i64)> {
    let [width, height] = size;
    let [pad_width, pad_height] = pad_size;

    if width > pad_width || height > pad_height {
        return None;
    }

    let (x, y) = match position {
        Position::Top => ((pad_width - width) / 2, 0),
        Position::Bottom => ((pad_width - width) / 2, pad_height - height),
        Position::Left => (0, (pad_height - height) / 2),
        Position::Right => (pad_width - width, (pad_height - height) / 2),
        Position::TopLeft => (0, 0),
        Position::TopRight => (pad_width - width, 0),
        Position::BottomLeft => (0, pad_height - height),
        Position::BottomRight => (pad_width - width, pad_height - height),
        Position::Center => ((pad_width - width) / 2, (pad_height - height) / 2),
    };

    Some((x.as_(), y.as_()))
}

pub trait Padding<P, S>
where
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    fn padding(
        self,
        pad_size: [u32; 2],
        position: &Position,
        color: P,
    ) -> (ImageBuffer<P, Vec<S>>, (u32, u32));

    fn padding_square(self, color: P) -> (ImageBuffer<P, Vec<S>>, (u32, u32));

    fn to_position(&self, pad_size: [u32; 2], position: &Position) -> Option<(i64, i64)>;

    fn to_position_square(&self) -> Option<((i64, i64), (u32, u32))>;
}

impl<P, S> Padding<P, S> for ImageBuffer<P, Vec<S>>
where
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    fn padding(
        self,
        pad_size: [u32; 2],
        position: &Position,
        color: P,
    ) -> (ImageBuffer<P, Vec<S>>, (u32, u32)) {
        self.to_position(pad_size, position)
            .map(|(x, y)| {
                let mut canvas = ImageBuffer::from_pixel(pad_size[0], pad_size[1], color);
                imageops::overlay(&mut canvas, &self, x, y);
                (canvas, (x as u32, y as u32))
            })
            .unwrap_or_else(|| (self, (0, 0)))
    }

    fn padding_square(self, color: P) -> (ImageBuffer<P, Vec<S>>, (u32, u32)) {
        if let Some((_, pad_size)) = self.to_position_square() {
            self.padding([pad_size.0, pad_size.1], &Position::Center, color)
        } else {
            (self, (0, 0))
        }
    }

    fn to_position(&self, pad_size: [u32; 2], position: &Position) -> Option<(i64, i64)> {
        let (width, height) = self.dimensions();

        to_position([width, height], pad_size, position)
    }

    fn to_position_square(&self) -> Option<((i64, i64), (u32, u32))> {
        let (width, height) = self.dimensions();

        let (pad_width, pad_height) = if width > height {
            (width, width)
        } else {
            (height, height)
        };

        self.to_position([pad_width, pad_height], &Position::Center)
            .map(|(x, y)| ((x, y), (pad_width, pad_height)))
    }
}
