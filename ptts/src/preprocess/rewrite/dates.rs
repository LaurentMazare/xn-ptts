//! Numeric dates: "20/12/2015" becomes "20-12 2015".

use super::{digits, suffix};
use crate::preprocess::Lang;

/// A date written `D/M` or `D/M/Y`, in either day-month order, with the slashes read as a pause:
/// "20/12/2015" becomes "20-12 2015", and "1/5" becomes "1-5". The same in every language.
pub(super) fn dates(word: &str, _lang: Lang) -> Option<String> {
    let (first, rest) = digits(word, 1, 2)?;
    let (second, rest) = digits(rest.strip_prefix('/')?, 1, 2)?;
    let (year, rest) = match rest.strip_prefix('/').and_then(|rest| digits(rest, 2, 4)) {
        Some((year, rest)) => (Some(year), rest),
        None => (None, rest),
    };
    let suffix = suffix(rest)?;
    match year {
        None => Some(format!("{first}-{second}{suffix}")),
        Some(year) => Some(format!("{first}-{second} {year}{suffix}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_are_read_out() {
        let cases = [
            ("1/5/2011...", "1-5 2011..."),
            ("16/01/1980", "16-01 1980"),
            ("12/31/2020.", "12-31 2020."),
            ("20/12/15", "20-12 15"),
            ("1/5.", "1-5."),
            ("1/5", "1-5"),
        ];
        for (input, expected) in cases {
            assert_eq!(dates(input, Lang::En).as_deref(), Some(expected), "{input:?}");
        }
        for input in ["1/5/", "1/5/2", "1/5/20111", "123/5", "1/", "/5", "1-5", "1/5/2011/"] {
            assert_eq!(dates(input, Lang::En), None, "{input:?}");
        }
    }
}
