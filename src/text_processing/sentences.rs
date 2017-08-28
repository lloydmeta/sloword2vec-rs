use regex::Regex;

use std::collections::HashSet;

lazy_static! {
    static ref ALPHANUM_SPACE: Regex = {
        Regex::new(r"[\w\s://]").unwrap()
    };
    static ref ALPHANUM: Regex = {
        Regex::new(r"[\w]").unwrap()
    };

    static ref EN_STOPWORDS: HashSet<&'static str> = {
        // Taken from https://github.com/6/stopwords-json/blob/master/dist/en.json
        let list = vec!["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"];
        let mut set = HashSet::with_capacity(list.len());
        for w in list {
            set.insert(w);
        }
        set
    };
}

pub(crate) struct Sentences {
    pub(crate) data: Vec<Vec<String>>,
}

impl Sentences {
    pub(crate) fn new(s: &str) -> Sentences {
        let sentences = s.split(".");
        let cleaned = sentences.map(remove_non_alpha);
        let data: Vec<Vec<String>> = cleaned
            .filter_map(|sentence| {
                let v: Vec<_> = sentence
                    .split(" ")
                    .filter_map(trim_to_none)
                    .filter(|s| not_stop_word(s))
                    .map(|s| s.to_string())
                    .collect();
                if v.len() > 0 { Some(v) } else { None }
            })
            .collect();
        Sentences { data }
    }

    #[cfg(test)]
    pub(crate) fn data(&self) -> Vec<Vec<&str>> {
        self.data
            .iter()
            .map(|v| v.iter().map(|w| w.as_ref()).collect::<Vec<_>>())
            .collect()
    }
}


pub(crate) fn not_stop_word(s: &str) -> bool {
    let lowered = s.to_lowercase();
    !EN_STOPWORDS.contains(&lowered[..])
}

pub(crate) fn trim_to_none(s: &str) -> Option<&str> {
    let trimmed = s.trim();
    if trimmed.len() > 0 {
        Some(trimmed)
    } else {
        None
    }
}

pub(crate) fn remove_non_alpha(s: &str) -> String {
    let chars: Vec<_> = s.chars().collect();
    let s1 = s.chars().into_iter();
    let v = s1.enumerate().filter_map(
        |(i, c)| if ALPHANUM_SPACE.is_match(
            &c.to_string(),
        ) && !c.is_whitespace()
        {
            Some(c)
        } else if
        // Check if the non-alpha space char is in the middle of 2 alphanums
        i > 0 && i < chars.len() - 1 && !c.is_whitespace() &&
                   !chars[i + 1].is_whitespace() &&
                   !chars[i - 1].is_whitespace() &&
                   ALPHANUM.is_match(&chars[i + 1].to_string()) &&
                   ALPHANUM.is_match(&chars[i - 1].to_string())
        {
            Some(c)
        } else if c.is_whitespace() {
            Some(' ')
        } else {
            None
        },
    );
    v.into_iter().collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_non_alpha() {
        let original = r#"
This is the newest thing. http://google.com!"#;
        let expected = r#" This is the newest thing http://google.com"#;
        assert_eq!(remove_non_alpha(original), expected);
    }

    #[test]
    fn test_sentence_parsing() {
        let s = Sentences::new(
            "hello there. my name is lloyd. Joe is one of my most trusted advisors",
        );
        assert_eq!(
            s.data(),
            vec![vec!["lloyd"], vec!["Joe", "trusted", "advisors"]]
        );
    }

}
