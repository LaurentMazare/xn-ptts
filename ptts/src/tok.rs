//! A [`crate::Tokenizer`] backed by Hugging Face [`tokenizers`].
//!
//! Each checkpoint supplies its own `tokenizer.json`. No tokenizer is bundled or guessed.
//!
//! Available with the `hf` feature.

/// A Hugging Face tokenizer.
pub struct Tok(tokenizers::Tokenizer);

impl Tok {
    /// Opens the checkpoint's explicitly supplied Hugging Face tokenizer JSON.
    pub fn open(path: &std::path::Path) -> xn::Result<Self> {
        tracing::info!(?path, "loading Hugging Face tokenizer");
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| xn::Error::wrap(e).with_path(path))?;
        Ok(Tok(tok))
    }

    /// Loads the contents of a `tokenizer.json`, for callers with no filesystem to read it from
    /// (the wasm demo fetches it over the network).
    pub fn from_bytes(json: &[u8]) -> xn::Result<Self> {
        let tok = tokenizers::Tokenizer::from_bytes(json).map_err(xn::Error::wrap)?;
        Ok(Tok(tok))
    }
}

impl crate::Tokenizer for Tok {
    fn encode(&self, text: &str) -> xn::Result<Vec<u32>> {
        let encoded = self.0.encode(text, false).map_err(xn::Error::wrap)?;
        Ok(encoded.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> xn::Result<String> {
        self.0.decode(ids, true).map_err(xn::Error::wrap)
    }
}

#[cfg(test)]
mod tests {
    const MINIMAL: &str = r#"{"version":"1.0","added_tokens":[],
      "model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["ab",-1.0],["c",-2.0]]}}"#;

    #[test]
    fn from_bytes_reads_a_tokenizer_json() {
        use crate::Tokenizer as _;
        let tok = super::Tok::from_bytes(MINIMAL.as_bytes()).unwrap();
        assert_eq!(tok.encode("abc").unwrap(), [1, 2]);
    }
}
