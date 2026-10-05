//! Phone numbers: "+16502349653" becomes "plus 1 6-5-zero 2-3-4 9-6-5-3".

use super::{digits, suffix};
use crate::preprocess::Lang;

/// A phone number written as one run of digits, grouped the way it is read out: international
/// numbers by their country code, local ones by the conventions of `lang`. Within a group
/// separated by dashes each digit is read on its own.
///
/// International numbers start with `+`, a known country code and at least five more digits.
/// Local numbers start with `0`, and are recognized in French (10 digits), English (11 digits,
/// the UK) and German (10 or 11 digits).
pub(super) fn phones(word: &str, lang: Lang) -> Option<String> {
    let (international, rest) = match word.strip_prefix('+') {
        Some(rest) => (true, rest),
        None => (false, word),
    };
    let (number, rest) = digits(rest, 1, usize::MAX)?;
    let suffix = suffix(rest)?;
    let spoken = if international {
        if number.len() < 6 {
            return None;
        }
        let (code, number) = country_code(number)?;
        let number = match code {
            "1" => group(number, &[3, 3, 4], 3, true, lang),
            "34" | "351" => group(number, &[3, 3, 3], 3, true, lang),
            "44" => uk(number, lang),
            "49" => de(number, lang),
            // French numbers among them: in pairs, read as numbers in French and digit by digit
            // in the other languages.
            _ => group(number, &[2 - number.len() % 2], 2, lang != Lang::Fr, lang),
        };
        // The `+` is spoken, as everywhere else: the tokenizer never sees one.
        format!("{} {code} {number}", lang.special_chars().plus)
    } else {
        if !number.starts_with('0') {
            return None;
        }
        match (lang, number.len()) {
            (Lang::Fr, 10) => group(number, &[], 2, false, lang),
            (Lang::En, 11) => uk(number, lang),
            (Lang::De, 10 | 11) => de(number, lang),
            _ => return None,
        }
    };
    Some(format!("{spoken}{suffix}"))
}

/// UK numbers: a London number is 2 + 4 + 4 digits after the leading zero, any other 3 + 3 + 4.
fn uk(number: &str, lang: Lang) -> String {
    let zero = usize::from(number.starts_with('0'));
    if number[zero..].starts_with('2') {
        group(number, &[2 + zero, 4, 4], 4, true, lang)
    } else {
        group(number, &[3 + zero, 3, 4], 4, true, lang)
    }
}

/// German numbers: an area code of 2 or 3 digits after the leading zero, then pairs.
fn de(number: &str, lang: Lang) -> String {
    let zero = usize::from(number.starts_with('0'));
    let national = &number[zero..];
    if ["15", "16", "17"].iter().any(|p| national.starts_with(p)) {
        group(number, &[2 + zero, 1, 1], 2, true, lang)
    } else if ["30", "40", "69"].iter().any(|p| national.starts_with(p)) {
        group(number, &[2 + zero], 2, true, lang)
    } else {
        group(number, &[3 + zero], 2, true, lang)
    }
}

/// The digits of `number` in groups of the lengths in `lens`, then of `then` digits until the
/// end. With `dashed`, the digits of a group are joined by dashes so that each is read on its own,
/// and in English a dashed zero is spelled "zero" so that it is not read "nil".
fn group(number: &str, lens: &[usize], then: usize, dashed: bool, lang: Lang) -> String {
    let mut groups = vec![];
    let mut group = vec![];
    for (i, digit) in number.char_indices() {
        if dashed && lang == Lang::En && digit == '0' {
            group.push("zero");
        } else {
            group.push(&number[i..=i]);
        }
        if group.len() >= lens.get(groups.len()).copied().unwrap_or(then) {
            groups.push(group.join(if dashed { "-" } else { "" }));
            group.clear();
        }
    }
    if !group.is_empty() {
        groups.push(group.join(if dashed { "-" } else { "" }));
    }
    groups.join(" ")
}

/// The country code `number` starts with, and the rest of it.
fn country_code(number: &str) -> Option<(&str, &str)> {
    let codes: [&[&str]; 3] = [&CODES_1, &CODES_2, &CODES_3];
    codes.into_iter().enumerate().find_map(|(i, codes)| {
        let (code, rest) = number.split_at_checked(i + 1)?;
        codes.contains(&code).then_some((code, rest))
    })
}

const CODES_1: [&str; 2] = ["1", "7"];

const CODES_2: [&str; 44] = [
    "20", "27", "30", "31", "32", "33", "34", "36", "39", "40", "41", "43", "44", "45", "46", "47",
    "48", "49", "51", "52", "53", "54", "55", "56", "57", "58", "60", "61", "62", "63", "64", "65",
    "66", "81", "82", "84", "86", "90", "91", "92", "93", "94", "95", "98",
];

const CODES_3: [&str; 149] = [
    "211", "212", "213", "216", "218", "220", "221", "222", "223", "224", "225", "226", "227",
    "228", "229", "230", "231", "232", "233", "234", "235", "236", "237", "238", "239", "240",
    "241", "242", "243", "244", "245", "248", "249", "250", "251", "252", "253", "254", "255",
    "256", "257", "258", "260", "261", "262", "263", "264", "265", "267", "268", "269", "291",
    "297", "298", "350", "351", "352", "353", "354", "355", "356", "357", "358", "359", "370",
    "371", "372", "374", "375", "376", "377", "378", "379", "380", "381", "382", "383", "385",
    "386", "387", "389", "420", "421", "423", "500", "501", "502", "503", "504", "505", "506",
    "507", "509", "591", "592", "593", "594", "595", "597", "598", "599", "670", "673", "674",
    "675", "676", "677", "678", "679", "680", "682", "683", "685", "686", "687", "688", "689",
    "691", "692", "850", "852", "853", "855", "856", "880", "886", "960", "961", "962", "963",
    "964", "965", "966", "967", "968", "970", "971", "972", "973", "974", "975", "976", "977",
    "992", "993", "994", "995", "996", "998",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_grouped() {
        let cases = [
            // Local numbers.
            (Lang::Fr, "0123456789", "01 23 45 67 89"),
            (Lang::Fr, "0123456789...", "01 23 45 67 89..."),
            (Lang::En, "07596854413", "zero-7-5-9 6-8-5 4-4-1-3"),
            (Lang::En, "01511234567", "zero-1-5-1 1-2-3 4-5-6-7"),
            (Lang::En, "02079460852", "zero-2-zero 7-9-4-6 zero-8-5-2"),
            (Lang::De, "01511234567", "0-1-5 1 1 2-3 4-5 6-7"),
            (Lang::De, "0301234567", "0-3-0 1-2 3-4 5-6 7"),
            // International numbers.
            (Lang::En, "+442079460852", "plus 44 2-zero 7-9-4-6 zero-8-5-2"),
            (Lang::Fr, "+330556791936", "plus 33 05 56 79 19 36"),
            (Lang::En, "+330556791936", "plus 33 zero-5 5-6 7-9 1-9 3-6"),
            (Lang::Fr, "+33556791936", "plus 33 5 56 79 19 36"),
            (Lang::En, "+33556791936", "plus 33 5 5-6 7-9 1-9 3-6"),
            (Lang::En, "+16502349653", "plus 1 6-5-zero 2-3-4 9-6-5-3"),
            (Lang::En, "+447700900123", "plus 44 7-7-zero zero-9-zero zero-1-2-3"),
            (Lang::En, "+442000900123", "plus 44 2-zero zero-zero-9-zero zero-1-2-3"),
            (Lang::De, "+491511234567", "Plus 49 1-5 1 1 2-3 4-5 6-7"),
            (Lang::De, "+9491511234567", "Plus 94 9 1-5 1-1 2-3 4-5 6-7"),
            (Lang::De, "+33651150652", "Plus 33 6 5-1 1-5 0-6 5-2"),
            (Lang::De, "+33651150652,", "Plus 33 6 5-1 1-5 0-6 5-2,"),
            (Lang::Es, "+34612345678", "mas 34 6-1-2 3-4-5 6-7-8"),
            (Lang::Pt, "+351912345678", "mais 351 9-1-2 3-4-5 6-7-8"),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(phones(input, lang).as_deref(), Some(expected), "{lang:?} {input:?}");
        }
        let declined = [
            (Lang::Fr, "1123456789"),
            // No such country code.
            (Lang::De, "+9901511234567"),
            // Too short.
            (Lang::De, "+1234"),
            // Local numbers only in French, English and German, at their lengths.
            (Lang::Es, "0123456789"),
            (Lang::Fr, "012345678"),
            (Lang::En, "0123456789"),
            (Lang::En, "+1-650-234-9653"),
            (Lang::En, "+"),
            (Lang::En, "0"),
        ];
        for (lang, input) in declined {
            assert_eq!(phones(input, lang), None, "{lang:?} {input:?}");
        }
    }
}
