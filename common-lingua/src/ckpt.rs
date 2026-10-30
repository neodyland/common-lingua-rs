use safetensors::{SafeTensorError, SafeTensors};
pub(crate) struct Checkpoint<'a>(SafeTensors<'a>);

pub type STResult<T> = Result<T, SafeTensorError>;

impl<'a> Checkpoint<'a> {
    pub fn from_bytes(b: &'a [u8]) -> STResult<Self> {
        Ok(Self(SafeTensors::<'a>::deserialize(b)?))
    }
    pub fn f32(&self, name: String) -> STResult<(Vec<f32>, Vec<usize>)> {
        let scale = self.0.tensor(&name)?;
        Ok((
            scale
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            scale.shape().to_vec(),
        ))
    }
    pub fn i8(&self, name: String) -> STResult<(Vec<i8>, Vec<usize>)> {
        let scale = self.0.tensor(&name)?;
        Ok((
            scale
                .data()
                .iter()
                .map(|c| i8::from_le_bytes([*c]).max(-127))
                .collect(),
            scale.shape().to_vec(),
        ))
    }
}
