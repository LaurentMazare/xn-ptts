//! Text normalization applied before tokenization.
//!
//! The model is trained on ordinary written prose, so characters it never saw -- typographic
//! quotes, bullets, arrows, emoji -- are best removed rather than tokenized, and symbols that
//! are read aloud (`@`, `+`, `=`) are best spelled out in the target language. [`normalize_text`]
//! does both, driven by a [`Lang`].
//!
//! Every caller says which language, or says not to normalize: [`Normalize`] is a required
//! argument to [`crate::synth::SynthBuilder::new`], and every frontend takes it as a required
//! flag. There is no default, deliberately. Output is noticeably better with normalization than
//! without, but normalizing German as English speaks `@` as "at" rather than "ät", so guessing
//! the language is worse than doing nothing.

mod rewrite;

pub use rewrite::{Rules, rewrite_word};

/// Spoken forms of the punctuation characters that are read aloud rather than dropped: `@`, `+`
/// and `=` anywhere in the text, and the separators inside emails, URLs and codes.
#[derive(Debug, Clone)]
pub struct SpecialChars {
    pub at: &'static str,
    pub plus: &'static str,
    pub equals: &'static str,
    pub colon: &'static str,
    pub slash: &'static str,
    pub dash: &'static str,
    pub dot: &'static str,
    pub underscore: &'static str,
}

pub const SPECIAL_CHARS_EN: SpecialChars = SpecialChars {
    at: "at",
    plus: "plus",
    equals: "equals",
    colon: "colon",
    slash: "slash",
    dash: "dash",
    dot: "dot",
    underscore: "underscore",
};

pub const SPECIAL_CHARS_FR: SpecialChars = SpecialChars {
    at: "arobaze",
    plus: "plus",
    equals: "égal",
    colon: "deux-points",
    slash: "slash",
    dash: "tiret",
    dot: "point",
    underscore: "underscore",
};

pub const SPECIAL_CHARS_DE: SpecialChars = SpecialChars {
    at: "ät",
    plus: "Plus",
    equals: "Gleich",
    colon: "Doppelpunkt",
    slash: "Slash",
    dash: "Bindestrich",
    dot: "Punkt",
    underscore: "Unterstrich",
};

pub const SPECIAL_CHARS_ES: SpecialChars = SpecialChars {
    at: "arroba",
    plus: "mas",
    equals: "igual",
    colon: "dos-puntos",
    slash: "slash",
    dash: "guion",
    dot: "punto",
    underscore: "guion bajo",
};

pub const SPECIAL_CHARS_PT: SpecialChars = SpecialChars {
    at: "arroba",
    plus: "mais",
    equals: "igual",
    colon: "dois-pontos",
    slash: "slash",
    dash: "hifen",
    dot: "ponto",
    underscore: "underscore",
};

/// Language driving the spoken forms used by [`normalize_text`].
///
/// Deliberately has no `Default`: the spoken forms differ per language, so a caller that has not
/// said which language it has is better off not normalizing at all. See [`Normalize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    En,
    Fr,
    De,
    Es,
    Pt,
}

impl std::str::FromStr for Lang {
    type Err = crate::Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "en" => Ok(Lang::En),
            "fr" => Ok(Lang::Fr),
            "de" => Ok(Lang::De),
            "es" => Ok(Lang::Es),
            "pt" => Ok(Lang::Pt),
            _ => Err(crate::Error::invalid_argument(format!(
                "unsupported language code: {s}; expected en, fr, de, es, pt, or none to skip \
                 normalization"
            ))),
        }
    }
}

impl Lang {
    /// The language code this variant parses from.
    pub fn as_str(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Fr => "fr",
            Lang::De => "de",
            Lang::Es => "es",
            Lang::Pt => "pt",
        }
    }

    pub fn special_chars(self) -> &'static SpecialChars {
        match self {
            Lang::En => &SPECIAL_CHARS_EN,
            Lang::Fr => &SPECIAL_CHARS_FR,
            Lang::De => &SPECIAL_CHARS_DE,
            Lang::Es => &SPECIAL_CHARS_ES,
            Lang::Pt => &SPECIAL_CHARS_PT,
        }
    }

    pub fn decimal_separator(self) -> &'static str {
        match self {
            Lang::En => "point",
            Lang::Fr => "virgule",
            Lang::De => "Komma",
            Lang::Es => "coma",
            Lang::Pt => "vírgula",
        }
    }
}

/// Whether to normalize, in which language, and with which [`Rules`].
///
/// There is no default and no "unset": [`crate::synth::SynthBuilder::new`] takes one of these,
/// so choosing is not something a caller can forget. [`Normalize::OFF`] is the way to say "hand
/// the text to the tokenizer as written", which is for callers that normalize it themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Normalize {
    lang: Option<Lang>,
    rules: Rules,
}

impl Normalize {
    /// Hand text to the tokenizer as written.
    pub const OFF: Self = Self { lang: None, rules: Rules::NONE };

    /// Normalize as `lang`, with the default rewrite rules, [`Rules::DEFAULT`].
    pub const fn for_lang(lang: Lang) -> Self {
        Self { lang: Some(lang), rules: Rules::DEFAULT }
    }

    /// This policy with `rules` instead. A no-op on [`Self::OFF`], which rewrites nothing.
    pub const fn with_rules(self, rules: Rules) -> Self {
        match self.lang {
            Some(lang) => Self { lang: Some(lang), rules },
            None => Self::OFF,
        }
    }

    /// Which rewrites this policy applies.
    pub const fn rules(self) -> Rules {
        self.rules
    }

    /// The language half of this policy, round-tripping through its `FromStr`. The rules are
    /// not part of it, since they parse from a flag of their own: see [`Rules`].
    pub fn as_str(self) -> &'static str {
        match self.lang {
            Some(lang) => lang.as_str(),
            None => "none",
        }
    }

    /// Normalize `text`, or hand it back untouched when this is [`Self::OFF`].
    ///
    /// Borrows when off, so opting out costs no allocation per request.
    pub fn apply<'a>(self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match self.lang {
            None => std::borrow::Cow::Borrowed(text),
            Some(lang) => std::borrow::Cow::Owned(normalize_text(text, lang, self.rules)),
        }
    }
}

/// Normalizes as `lang`, with the default rewrite rules: what [`Normalize::for_lang`] makes.
impl From<Lang> for Normalize {
    fn from(lang: Lang) -> Self {
        Self::for_lang(lang)
    }
}

/// Parse what the frontends' `--lang` flag accepts: a language code, or `none` / `off`. The
/// rules are a flag of their own, see [`Rules`].
impl std::str::FromStr for Normalize {
    type Err = crate::Error;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "none" | "off" => Ok(Self::OFF),
            other => other.parse().map(Self::for_lang),
        }
    }
}

/// Character sink that keeps the output free of the punctuation pile-ups the substitutions
/// below would otherwise produce: a `.` or `,` swallows any whitespace and punctuation
/// immediately before it, and runs of whitespace collapse to a single space.
struct StringAppender {
    buffer: Vec<char>,
}

impl StringAppender {
    fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    fn push(&mut self, c: char) {
        if c == '.' || c == ',' {
            while self
                .buffer
                .last()
                .is_some_and(|&l| l.is_whitespace() || (l.is_ascii_punctuation() && !self.keeps(l)))
            {
                self.buffer.pop();
            }
        }
        self.buffer.push(c);
    }

    fn push_str(&mut self, s: &str) {
        for c in s.chars() {
            self.push(c);
        }
    }

    /// Whether `last`, the last character pushed, stays before a `.` or `,`. Quotes do: dropping
    /// a closing one would leave the opening one unbalanced. So do the symbols the rewrite rules
    /// read: `@` and `+`, which are spelled out after the rules, and the `$` of an amount, as in
    /// "it costs 5$.".
    fn keeps(&self, last: char) -> bool {
        match last {
            '"' | '\'' | '@' | '+' => true,
            '$' => self.buffer.len() >= 2 && self.buffer[self.buffer.len() - 2].is_ascii_digit(),
            _ => false,
        }
    }

    /// `word` as a word of its own, for a symbol read aloud.
    fn push_spoken(&mut self, word: &str) {
        self.push_whitespace();
        self.push_str(word);
        self.push_whitespace();
    }

    fn into_string(mut self) -> String {
        self.pop_whitespace();
        self.buffer.into_iter().collect()
    }

    fn last_is_whitespace(&self) -> bool {
        self.buffer.last().is_some_and(|c| c.is_whitespace())
    }

    fn push_whitespace(&mut self) {
        if !self.last_is_whitespace() && !self.buffer.is_empty() {
            self.push(' ');
        }
    }

    fn pop_whitespace(&mut self) {
        while self.last_is_whitespace() {
            self.buffer.pop();
        }
    }
}

fn is_emoji(c: char) -> bool {
    let c = c as u32;
    matches!(c,
        0x1F600..=0x1FAFF |
        0x2600..=0x27BF |
        // Flags (regional indicator symbols)
        0x1F1E6..=0x1F1FF
    )
}

/// Unicode characters that read as a double quotation mark: guillemets,
/// including the single-angle `‹ ›` pair, which quotes speech in the same way
/// the double one does, the curly and reversed variants, the low-9 ones
/// sitting on the baseline (German/Czech opening quotes), double primes,
/// dingbat and CJK corner quotes, and the fullwidth form.
fn is_double_quote(c: char) -> bool {
    matches!(
        c,
        '»' | '«'
            | '‹'
            | '›'
            | '“'
            | '”'
            | '„'
            | '‟'
            | '″'
            | '‶'
            | '⹂'
            | '❝'
            | '❞'
            | '❠'
            | '〝'
            | '〞'
            | '〟'
            | '＂'
    )
}

/// Unicode characters that read as a single quotation mark or apostrophe:
/// the curly and reversed variants, the low-9 one sitting on the baseline,
/// primes, dingbat quotes, the fullwidth form and the modifier letter and
/// accent characters commonly typed in place of an apostrophe.
fn is_single_quote(c: char) -> bool {
    matches!(
        c,
        '‘' | '’'
            | '‚'
            | '‛'
            | '′'
            | '‵'
            | '❛'
            | '❜'
            | '❟'
            | '＇'
            | 'ʼ'
            | 'ʻ'
            | 'ʹ'
            | '´'
            | '`'
    )
}

/// Rewrite `input` into the character set the model was trained on.
///
/// Typographic quotes, dashes, bullets, arrows and emoji are dropped or folded to their ASCII
/// equivalents; `@`, `+` and `=` are spelled out in `lang`; `;`, parentheses and a `:` with
/// whitespace on either side become commas, which is how the model is asked to pause. A `:`
/// between two non-space characters, as in `10:30`, is kept.
///
/// Then each word goes to the `rules`, and `@` and `+` are spelled out only in the words no rule
/// claimed, since the email and phone rules read them.
pub fn normalize_text(input: &str, lang: Lang, rules: Rules) -> String {
    let mut res = StringAppender::new();
    let mut chars = input.chars().peekable();
    let mut prev = None;
    while let Some(c) = chars.next() {
        match c {
            c if is_double_quote(c) => res.push('"'),
            c if is_single_quote(c) => res.push('\''),
            '‐' | '‑' | '‒' | '―' => res.push('-'),
            // The two dashes below are not - (ascii 45) but similar unicode chars.
            '–' | '*' | '—' | '[' | ']' | '{' | '}' => res.push_whitespace(),
            '•' | '‣' | '◦' | '·' | '→' | '←' | '↑' | '↓' | '➡' | '➜' => {
                res.push_whitespace();
            }
            '…' => res.push('.'),
            '=' => res.push_spoken(lang.special_chars().equals),
            ':' if !prev.is_none_or(char::is_whitespace)
                && !chars.peek().is_none_or(|c| c.is_whitespace()) =>
            {
                res.push(':')
            }
            ';' | ':' | '(' | ')' => {
                res.pop_whitespace();
                res.push(',');
                res.push_whitespace();
            }
            c => {
                if is_emoji(c) || c.is_control() || c.is_whitespace() {
                    res.push_whitespace();
                } else {
                    res.push(c)
                }
            }
        }
        prev = Some(c);
    }
    let text = res.into_string();
    let words = text.split(' ').map(|w| match rewrite_word(w, lang, rules) {
        Some(rewritten) => rewritten,
        None => spell_symbols(w, lang, rules),
    });
    words.collect::<Vec<_>>().join(" ")
}

/// Spell out the `@` and `+` of a word no rule claimed, as the character pass spells `=`, and
/// give the pieces they leave their own turn at the rules: "1500+20" reads "1 thousand 500 plus
/// 20", as it would had the character pass spelled the `+`.
fn spell_symbols(word: &str, lang: Lang, rules: Rules) -> String {
    if !word.contains(['@', '+']) {
        return word.to_string();
    }
    let mut res = StringAppender::new();
    for c in word.chars() {
        match c {
            '@' => res.push_spoken(lang.special_chars().at),
            '+' => res.push_spoken(lang.special_chars().plus),
            c => res.push(c),
        }
    }
    let text = res.into_string();
    let words = text.split(' ').map(|w| rewrite_word(w, lang, rules).unwrap_or_else(|| w.into()));
    words.collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_text_cases() {
        let cases: &[(&str, &str)] = &[
            ("Hello, world!", "Hello, world!"),
            ("", ""),
            ("“hello” world it's", "\"hello\" world it's"),
            ("«hello» ‹world›", "\"hello\" \"world\""),
            ("a‐b‑c‒d―e", "a-b-c-d-e"),
            ("a–b—c", "a b c"),
            ("foo (bar) [baz] {qux} *quux*", "foo, bar, baz qux quux"),
            ("• ‣ ◦ · a→b←c↑d↓e ➡ ➜", "a b c d e"),
            ("wait… a…b", "wait. a.b"),
            ("user@host @home", "user at host at home"),
            ("café résumé 日本語", "café résumé 日本語"),
            ("hello 😀 flag 🇫🇷 sun ☀", "hello flag sun"),
            // ';', '(' / ')' and a ':' next to whitespace collapse to ", " (comma + single
            // space); a ':' between two non-space characters is kept.
            ("a;b:c", "a, b:c"),
            ("time: 10:30", "time, 10:30"),
            ("a :b", "a, b"),
            ("note:", "note,"),
            (":start", ", start"),
            ("; leading", ", leading"),
            ("hello (world)", "hello, world,"),
            // Surrounding whitespace is absorbed into the comma replacement.
            ("foo ; bar  :  baz", "foo, bar, baz"),
            // Runs of ASCII and non-ASCII whitespace collapse to a single space,
            // and trailing whitespace is stripped.
            ("a   b\t\tc\n\nd", "a b c d"),
            ("hello   ", "hello"),
            ("a • b • c", "a b c"),
            (
                "“Hello”; please email user@host (now)… 🚀",
                "\"Hello\", please email user at host, now.",
            ),
            (
                "Numbers: one, two, three, four, five. Special items: at sign, hash, dollar, percent.",
                "Numbers, one, two, three, four, five. Special items, at sign, hash, dollar, percent.",
            ),
            (
                "The conference will be held on Tuesday, March 15th at 3:30 PM.",
                "The conference will be held on Tuesday, March 15th at 3:30 PM.",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(
                &normalize_text(input, Lang::En, Rules::DEFAULT),
                expected,
                "input: {input:?}"
            );
        }
    }

    #[test]
    fn spoken_symbols_follow_the_language() {
        assert_eq!(normalize_text("a@b", Lang::En, Rules::DEFAULT), "a at b");
        assert_eq!(normalize_text("a@b", Lang::Fr, Rules::DEFAULT), "a arobaze b");
        assert_eq!(normalize_text("a@b", Lang::De, Rules::DEFAULT), "a ät b");
        assert_eq!(normalize_text("1+1=2", Lang::Es, Rules::DEFAULT), "1 mas 1 igual 2");
        assert_eq!(normalize_text("1+1=2", Lang::Pt, Rules::DEFAULT), "1 mais 1 igual 2");
    }

    /// `@` and `+` reach the rules, which read them in emails and phone numbers, and are
    /// spelled out as before everywhere else, even with no rules at all.
    #[test]
    fn symbols_are_spelled_after_the_rules() {
        let cases = [
            ("Write to laurent.mazare@gmail.com!", "Write to laurent dot mazare at gmail dot com!"),
            ("Mail (laurent+tag@gmail.com)", "Mail, laurent plus tag at gmail dot com,"),
            ("user@host @home", "user at host at home"),
            ("a@.", "a at."),
            ("x @, y", "x at, y"),
            ("C++.", "C plus plus."),
            ("1500+20", "1 thousand 500 plus 20"),
            ("+33612345678", "plus 33 billion 612 million 345 thousand 678"),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize_text(input, Lang::En, Rules::DEFAULT), expected, "{input:?}");
        }
        assert_eq!(normalize_text("foo@bar.com", Lang::En, Rules::NONE), "foo at bar.com");
        assert_eq!(normalize_text("C++.", Lang::En, Rules::NONE), "C plus plus.");
        let phones = "phones".parse().unwrap();
        assert_eq!(normalize_text("+33612345678.", Lang::Fr, phones), "plus 33 6 12 34 56 78.");
    }

    #[test]
    fn sentences_are_rewritten() {
        let cases = [
            (Lang::En, "It costs $1500 today.", "It costs 1 thousand 500 dollars today."),
            (Lang::En, "It costs 5$.", "It costs 5 dollars."),
            (Lang::En, "It costs -$5.", "It costs minus 5 dollars."),
            // A `$` that is not part of an amount goes before a period, as other symbols do.
            (Lang::En, "In $.", "In."),
            (Lang::En, "Only £50 left.", "Only 50 pounds left."),
            (Lang::Fr, "Ça coûte 500€ aujourd'hui.", "Ça coûte 500 euros aujourd'hui."),
            (Lang::De, "Es kostet $2500 heute.", "Es kostet 2 Tausend 500 Dollar heute."),
            (
                Lang::En,
                "Call 555-123-4567 before 2024-05-12, it costs 12345678.",
                "Call 555 123 4 5 6 7 before 2024-05-12, it costs 12 million 345 thousand 678.",
            ),
            (Lang::En, "Visit www.kyutai.fr today.", "Visit W-W-W dot kyutai dot F-R today."),
            (
                Lang::Fr,
                "Voir https://www.kyutai.fr/.",
                "Voir H-T-T-P-S deux-points slash slash W-W-W point kyutai point F-R.",
            ),
            // Times and dates are opt-in.
            (Lang::En, "It's 1:06pm on 20/12/2015.", "It's 1:06pm on 20/12/2015."),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(normalize_text(input, lang, Rules::DEFAULT), expected, "{input:?}");
        }
        assert_eq!(
            normalize_text("It's 1:06pm on 20/12/2015.", Lang::En, Rules::ALL),
            "It's 1-06 PM on 20-12 2015."
        );
        assert_eq!(normalize_text("Um 08:20 Uhr.", Lang::De, Rules::ALL), "Um 8 Uhr 20 Uhr.");
    }

    /// The frontends take one flag for the language and for turning
    /// normalization off, so both spellings of "off" have to parse rather than
    /// error, and every policy has to spell itself back.
    #[test]
    fn normalize_parses_and_round_trips() {
        for lang in [Lang::En, Lang::Fr, Lang::De, Lang::Es, Lang::Pt] {
            let norm = Normalize::for_lang(lang);
            assert_eq!(Normalize::from(lang), norm);
            assert_eq!(lang.as_str().parse::<Normalize>().unwrap(), norm);
            assert_eq!(norm.as_str().parse::<Normalize>().unwrap(), norm);
        }
        assert_eq!("EN".parse::<Normalize>().unwrap(), Normalize::for_lang(Lang::En));
        assert_eq!(" en ".parse::<Normalize>().unwrap(), Normalize::for_lang(Lang::En));
        assert_eq!(" Off ".parse::<Normalize>().unwrap(), Normalize::OFF);
        assert_eq!("none".parse::<Normalize>().unwrap(), Normalize::OFF);
        assert_eq!("off".parse::<Normalize>().unwrap(), Normalize::OFF);
        assert_eq!(Normalize::OFF.as_str().parse::<Normalize>().unwrap(), Normalize::OFF);
        let err = "klingon".parse::<Normalize>().unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("klingon"), "{msg}");
        assert!(msg.contains("none"), "the error must name the opt-out: {msg}");
    }

    /// Opting out has to hand the text through untouched, and without
    /// allocating: `apply` runs on every request.
    #[test]
    fn apply_follows_the_policy() {
        use std::borrow::Cow;
        assert_eq!(Normalize::OFF.apply("a@b (c)"), "a@b (c)");
        assert_eq!(Normalize::for_lang(Lang::En).apply("a@b (c)"), "a at b, c,");
        assert_eq!(Normalize::for_lang(Lang::Fr).apply("a@b"), "a arobaze b");
        assert!(matches!(Normalize::OFF.apply("text"), Cow::Borrowed(_)));
        // The rules travel with the policy, so `apply` is all a caller needs.
        let en = Normalize::for_lang(Lang::En);
        assert_eq!(en.apply("I paid 1500."), "I paid 1 thousand 500.");
        assert_eq!(en.with_rules(Rules::NONE).apply("I paid 1500."), "I paid 1500.");
    }

    #[test]
    fn lang_round_trips_through_str() {
        use std::str::FromStr;
        for (s, lang) in [
            ("en", Lang::En),
            ("FR", Lang::Fr),
            ("de", Lang::De),
            ("es", Lang::Es),
            ("pt", Lang::Pt),
        ] {
            assert_eq!(Lang::from_str(s).unwrap(), lang);
        }
        assert!(Lang::from_str("klingon").is_err());
    }
}
