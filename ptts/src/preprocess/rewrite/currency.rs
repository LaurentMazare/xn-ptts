//! Amounts of money: "$1,234.56" becomes "1 thousand 234 dollars and 56 cents".

use super::numbers::number_words;
use super::{all_digits, split_suffix};
use crate::preprocess::Lang;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Currency {
    Dollar,
    Euro,
    Pound,
}

impl Currency {
    const SYMBOLS: [char; 3] = ['$', '€', '£'];

    fn from_symbol(symbol: char) -> Option<Self> {
        match symbol {
            '$' => Some(Self::Dollar),
            '€' => Some(Self::Euro),
            '£' => Some(Self::Pound),
            _ => None,
        }
    }

    /// The currency's name in `lang`, singular and plural.
    fn name(self, lang: Lang) -> (&'static str, &'static str) {
        match (self, lang) {
            (Self::Dollar, Lang::En | Lang::Fr) => ("dollar", "dollars"),
            (Self::Dollar, Lang::De) => ("Dollar", "Dollar"),
            (Self::Dollar, Lang::Es | Lang::Pt) => ("dólar", "dólares"),
            (Self::Euro, Lang::De) => ("Euro", "Euro"),
            (Self::Euro, _) => ("euro", "euros"),
            (Self::Pound, Lang::En) => ("pound", "pounds"),
            (Self::Pound, Lang::Fr) => ("livre", "livres"),
            (Self::Pound, Lang::De) => ("Pfund", "Pfund"),
            (Self::Pound, Lang::Es | Lang::Pt) => ("libra", "libras"),
        }
    }
}

/// A number with a currency symbol before or after it, read as [`super::numbers`] reads the
/// number, followed by the currency's name. In English and French an amount with exactly two
/// decimals is read in cents: "$4.59" becomes "4 dollars and 59 cents", "4,59€" becomes "4 euros
/// et 59 centimes".
pub(super) fn currency(word: &str, lang: Lang) -> Option<String> {
    let (body, suffix) = split_suffix(word);
    // "-$500" reads as "$-500" does.
    if let Some(body) = body.strip_prefix('-')
        && let Some(amount) = body.strip_prefix(Currency::SYMBOLS)
    {
        let symbol = &body[..body.len() - amount.len()];
        return currency(&format!("{symbol}-{amount}{suffix}"), lang);
    }
    // The symbol leads or trails, never both: "$5€" is not an amount.
    let (symbol, amount) = match body.strip_prefix(Currency::SYMBOLS) {
        Some(amount) => (body.chars().next()?, amount),
        None => (body.chars().next_back()?, body.strip_suffix(Currency::SYMBOLS)?),
    };
    let currency = Currency::from_symbol(symbol)?;
    if let Some(spoken) = with_cents(amount, currency, lang) {
        return Some(format!("{spoken}{suffix}"));
    }
    let (amount, _) = number_words(amount, lang)?;
    // Always the plural, even for 1: "$1" reads "1 dollars". A quirk, kept so that the readings
    // stay identical to the ones the serving stack already speaks.
    let (_, name) = currency.name(lang);
    Some(format!("{}{suffix}", with_name(&amount, name, lang)))
}

/// `amount` read as whole units and cents, when it has exactly two decimals: English writes it
/// "1,234.56" and French "1234,56". `None` for any other amount, and in the other languages.
fn with_cents(amount: &str, currency: Currency, lang: Lang) -> Option<String> {
    let (whole, cents) = match lang {
        Lang::En => amount.rsplit_once('.').filter(|(whole, _)| {
            all_digits(whole)
                || whole.split(',').enumerate().all(|(i, group)| {
                    all_digits(group) && if i == 0 { group.len() <= 3 } else { group.len() == 3 }
                })
        })?,
        Lang::Fr => amount.rsplit_once(',').filter(|(whole, _)| all_digits(whole))?,
        Lang::De | Lang::Es | Lang::Pt => return None,
    };
    if cents.len() != 2 || !all_digits(cents) {
        return None;
    }
    let (whole, _) = number_words(whole, lang)?;
    let (one, many) = currency.name(lang);
    let units = with_name(&whole, if whole == "1" { one } else { many }, lang);
    if cents == "00" {
        return Some(units);
    }
    // "05" is read "5". The subdivision is always plural, and English says cents even for
    // pounds: quirks again, kept for the same reason as the plural above.
    let cents = cents.strip_prefix('0').unwrap_or(cents);
    let (and, cent) = match (lang, currency) {
        (Lang::Fr, Currency::Euro) => ("et", "centimes"),
        (Lang::Fr, _) => ("et", "cents"),
        _ => ("and", "cents"),
    };
    if whole == "0" {
        return Some(format!("{cents} {cent}"));
    }
    Some(format!("{units} {and} {cents} {cent}"))
}

/// `amount` followed by the currency's `name`. Scale words are nouns, so in French, Spanish and
/// Portuguese a currency right after one takes a linking "de" ("2 millions de dollars"), elided
/// before a vowel in French ("2 millions d'euros"). A numeral adjective takes none ("500 mille
/// euros").
fn with_name(amount: &str, name: &str, lang: Lang) -> String {
    let nouns: &[&str] = match lang {
        Lang::Fr => &["million", "millions", "milliard", "milliards"],
        Lang::Es => &["millón", "millones"],
        Lang::Pt => &["milhão", "milhões", "bilhão", "bilhões"],
        Lang::En | Lang::De => &[],
    };
    if !nouns.iter().any(|noun| amount.ends_with(noun)) {
        format!("{amount} {name}")
    } else if lang == Lang::Fr && name.starts_with(['a', 'e', 'i', 'o', 'u', 'é']) {
        format!("{amount} d'{name}")
    } else {
        format!("{amount} de {name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amounts_are_read_with_their_currency() {
        let cases = [
            // A symbol before the amount.
            (Lang::En, "$1234", "1 thousand 234 dollars"),
            (Lang::En, "$1,234.56", "1 thousand 234 dollars and 56 cents"),
            (Lang::En, "$1,234.56...", "1 thousand 234 dollars and 56 cents..."),
            (Lang::En, "$4.59", "4 dollars and 59 cents"),
            (Lang::En, "$0.01", "1 cents"),
            (Lang::En, "$0.10", "10 cents"),
            (Lang::En, "$1.00", "1 dollar"),
            (Lang::En, "$1.10", "1 dollar and 10 cents"),
            (Lang::En, "$3.14", "3 dollars and 14 cents"),
            (Lang::En, "$0.00", "0 dollars"),
            (Lang::En, "$1", "1 dollars"),
            (Lang::En, "£1.50", "1 pound and 50 cents"),
            (Lang::Fr, "$4,59", "4 dollars et 59 cents"),
            (Lang::Fr, "$0,01", "1 cents"),
            (Lang::Fr, "$1,00", "1 dollar"),
            (Lang::Fr, "$1,10", "1 dollar et 10 cents"),
            (Lang::Fr, "$0,00", "0 dollars"),
            (Lang::Fr, "$100", "100 dollars"),
            (Lang::De, "$2500", "2 Tausend 500 Dollar"),
            (Lang::Es, "$50", "50 dólares"),
            (Lang::Pt, "$50!", "50 dólares!"),
            (Lang::En, "€500...", "500 euros..."),
            (Lang::En, "€0", "0 euros"),
            (Lang::En, "€0.5", "0 point 5 euros"),
            (Lang::Fr, "€1000", "mille euros"),
            (Lang::De, "€1000", "ein Tausend Euro"),
            (Lang::De, "$1000000", "eine Million Dollar"),
            (Lang::En, "£250", "250 pounds"),
            (Lang::Fr, "£250", "250 livres"),
            (Lang::De, "£250", "250 Pfund"),
            (Lang::Es, "£250", "250 libras"),
            // A symbol after it.
            (Lang::En, "1234$", "1 thousand 234 dollars"),
            (Lang::Fr, "4,59€", "4 euros et 59 centimes"),
            (Lang::Fr, "0,01€", "1 centimes"),
            (Lang::Fr, "1,00€", "1 euro"),
            (Lang::Fr, "0,00€", "0 euros"),
            (Lang::Fr, "500€!", "500 euros!"),
            (Lang::De, "100£", "100 Pfund"),
            (Lang::De, "1000000000€", "eine Milliarde Euro"),
            // Only English and French read cents: elsewhere two decimals are a plain number.
            (Lang::De, "1,23$", "1 Komma 23 Dollar"),
            (Lang::De, "1,23$!!!", "1 Komma 23 Dollar!!!"),
            // Signs, including a zero integer part.
            (Lang::En, "$-500", "minus 500 dollars"),
            (Lang::En, "-$500.", "minus 500 dollars."),
            (Lang::De, "-€1.000", "minus ein Tausend Euro"),
            (Lang::Fr, "-0,50€", "moins 0 virgule 50 euros"),
            // A linking "de" after a scale noun, elided before a vowel in French, and none after
            // the numeral adjective "mille".
            (Lang::Fr, "€2000000", "2 millions d'euros"),
            (Lang::Fr, "$2000000", "2 millions de dollars"),
            (Lang::Fr, "£3000000000.", "3 milliards de livres."),
            (Lang::Fr, "€2000000,00", "2 millions d'euros"),
            (Lang::Fr, "2000000,50€", "2 millions d'euros et 50 centimes"),
            (Lang::Fr, "€2500000", "2 millions 500 mille euros"),
            (Lang::Es, "€2000000", "2 millones de euros"),
            (Lang::Pt, "$2000000", "2 milhões de dólares"),
            (Lang::En, "$2000000", "2 million dollars"),
            (Lang::De, "$2000000", "2 Millionen Dollar"),
        ];
        for (lang, input, expected) in cases {
            assert_eq!(currency(input, lang).as_deref(), Some(expected), "{lang:?} {input:?}");
        }
        for input in ["1234", "$abc", "$", "$5€", "5$5", "$007", "$1.2.3", "$ 5", "-$", "--$5"] {
            assert_eq!(currency(input, Lang::En), None, "{input:?}");
        }
    }
}
