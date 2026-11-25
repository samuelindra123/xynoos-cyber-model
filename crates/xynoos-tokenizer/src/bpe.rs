use crate::{
    BASE_VOCABULARY, BYTE_VALUES, Result, TokenId, Tokenizer, TokenizerError, TokenizerLimits, add,
    allocate, bytes_used, charge, check_capacity, mul, reserve,
};

use TokenizerError::*;

type Pair = (TokenId, TokenId);

#[derive(Debug)]
pub struct Merge {
    pub(crate) pair: Pair,
    pub(crate) expansion: Vec<u8>,
}

impl Merge {
    pub fn pair(&self) -> Pair {
        self.pair
    }

    pub fn expansion(&self) -> &[u8] {
        &self.expansion
    }
}

#[derive(Debug)]
pub struct BpeModel {
    merges: Vec<Merge>,
    // Sorted by operand IDs. The value is the merge rank.
    lookup: Vec<(Pair, usize)>,
    expansion_capacity: usize,
}

impl BpeModel {
    pub(crate) fn new(count: usize, buffer_bytes: usize) -> Result<Self> {
        check_capacity(
            add(mul::<Merge>(count)?, mul::<(Pair, usize)>(count)?)?,
            buffer_bytes,
        )?;
        let model = Self {
            merges: allocate(count)?,
            lookup: allocate(count)?,
            expansion_capacity: 0,
        };
        check_capacity(model.retained_bytes()?, buffer_bytes)?;
        Ok(model)
    }

    pub fn merges(&self) -> &[Merge] {
        &self.merges
    }

    pub fn retained_bytes(&self) -> Result<usize> {
        add(
            add(bytes_used(&self.merges)?, bytes_used(&self.lookup)?)?,
            self.expansion_capacity,
        )
    }

    fn expansion(&self, id: TokenId) -> Result<&[u8]> {
        match id {
            0..=255 => Ok(&BYTE_VALUES[id as usize..=id as usize]),
            256..=259 => Err(MalformedArtifact),
            _ => self
                .merges
                .get(id as usize - BASE_VOCABULARY)
                .map(Merge::expansion)
                .ok_or(MalformedArtifact),
        }
    }

    pub(crate) fn append(
        &mut self,
        pair: Pair,
        hex: Option<&[u8]>,
        expansion_limit: usize,
        buffer_limit: usize,
    ) -> Result<()> {
        let position = self
            .lookup
            .binary_search_by_key(&pair, |&(key, _)| key)
            .err()
            .ok_or(MalformedArtifact)?;
        let left = self.expansion(pair.0)?;
        let right = self.expansion(pair.1)?;
        let size = add(left.len(), right.len())?;
        if size > expansion_limit {
            return Err(ExpansionLimitExceeded);
        }
        if let Some(hex) = hex {
            if hex.len() != mul::<[u8; 2]>(size)? {
                return Err(MalformedArtifact);
            }
            for (&byte, digits) in left.iter().chain(right).zip(hex.as_chunks::<2>().0) {
                if *digits != crate::artifact::hex_pair(byte) {
                    return Err(MalformedArtifact);
                }
            }
        }
        let retained = self.retained_bytes()?;
        check_capacity(add(retained, size)?, buffer_limit)?;
        let mut expansion = allocate(size)?;
        let capacity = bytes_used(&expansion)?;
        check_capacity(add(retained, capacity)?, buffer_limit)?;
        expansion.extend(left.iter().chain(right).copied());
        self.expansion_capacity = add(self.expansion_capacity, capacity)?;
        self.lookup.insert(position, (pair, self.merges.len()));
        self.merges.push(Merge { pair, expansion });
        Ok(())
    }

    pub(crate) fn encode(&self, tokens: &mut Vec<TokenId>, mut work: usize) -> Result<()> {
        if self.lookup.is_empty() {
            return Ok(());
        }
        loop {
            let mut best = None;
            for window in tokens.windows(2) {
                charge(&mut work, 1)?;
                let pair = (window[0], window[1]);
                let mut first = 0;
                let mut end = self.lookup.len();
                while first < end {
                    charge(&mut work, 1)?;
                    let mid = first + (end - first) / 2;
                    let (key, rank) = self.lookup[mid];
                    match key.cmp(&pair) {
                        std::cmp::Ordering::Less => first = mid + 1,
                        std::cmp::Ordering::Greater => end = mid,
                        std::cmp::Ordering::Equal => {
                            best = Some(best.map_or(rank, |old: usize| old.min(rank)));
                            break;
                        }
                    }
                }
            }
            let Some(rank) = best else { return Ok(()) };
            charge(&mut work, tokens.len())?;
            replace(tokens, self.merges[rank].pair, merge_id(rank)?);
        }
    }
}

fn merge_id(rank: usize) -> Result<TokenId> {
    TokenId::try_from(add(BASE_VOCABULARY, rank)?).map_err(|_| ArithmeticOverflow)
}

fn replace(tokens: &mut Vec<TokenId>, pair: Pair, result: TokenId) {
    let mut read = 0;
    let mut write = 0;
    while read < tokens.len() {
        if tokens[read] == pair.0 && tokens.get(read + 1) == Some(&pair.1) {
            tokens[write] = result;
            read += 2;
        } else {
            tokens[write] = tokens[read];
            read += 1;
        }
        write += 1;
    }
    tokens.truncate(write);
}

/// Algorithm-only trainer for byte slices. Dataset authorization is enforced
/// by the governed fit API in xynoos-data.
pub struct BpeTrainer {
    limits: TokenizerLimits,
    requested_merges: usize,
    corpus_bytes: usize,
    sequence_capacity: usize,
    documents: Vec<Vec<TokenId>>,
    failure: Option<TokenizerError>,
}

impl BpeTrainer {
    pub fn new(requested_merges: usize, limits: TokenizerLimits) -> Result<Self> {
        limits.validate()?;
        if requested_merges > limits.merges {
            return Err(VocabularyLimitExceeded);
        }
        Ok(Self {
            limits,
            requested_merges,
            corpus_bytes: 0,
            sequence_capacity: 0,
            documents: Vec::new(),
            failure: None,
        })
    }

    pub fn push_document(&mut self, bytes: &[u8]) -> Result<()> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let result = self.push(bytes);
        self.failure = result.err();
        result
    }

    fn push(&mut self, bytes: &[u8]) -> Result<()> {
        if self.documents.len() == self.limits.corpus_documents {
            return Err(TooManyDocuments);
        }
        if bytes.len() > self.limits.input_bytes {
            return Err(InputTooLarge);
        }
        let total = add(self.corpus_bytes, bytes.len())?;
        if total > self.limits.corpus_bytes {
            return Err(CorpusTooLarge);
        }
        let docs = add(self.documents.len(), 1)?;
        let table = mul::<Vec<TokenId>>(self.documents.capacity().max(docs))?;
        let expected = add(self.sequence_capacity, mul::<TokenId>(bytes.len())?)?;
        check_capacity(add(table, expected)?, self.limits.buffer_bytes)?;
        if docs > self.documents.capacity() {
            let capacity = self
                .documents
                .capacity()
                .saturating_mul(2)
                .max(docs)
                .min(self.limits.corpus_documents);
            let growth =
                if add(mul::<Vec<TokenId>>(capacity)?, expected)? <= self.limits.buffer_bytes {
                    capacity
                } else {
                    docs
                };
            reserve(&mut self.documents, growth - (docs - 1))?;
        }
        check_capacity(
            add(bytes_used(&self.documents)?, expected)?,
            self.limits.buffer_bytes,
        )?;
        let mut document = allocate(bytes.len())?;
        let actual = add(self.sequence_capacity, bytes_used(&document)?)?;
        check_capacity(
            add(bytes_used(&self.documents)?, actual)?,
            self.limits.buffer_bytes,
        )?;
        document.extend(bytes.iter().map(|&byte| TokenId::from(byte)));
        self.documents.push(document);
        self.sequence_capacity = actual;
        self.corpus_bytes = total;
        Ok(())
    }

    pub fn finish(mut self) -> Result<Tokenizer> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let cap = self.limits.buffer_bytes;
        let corpus = add(bytes_used(&self.documents)?, self.sequence_capacity)?;
        if self.requested_merges == 0 || self.documents.iter().all(|doc| doc.len() < 2) {
            return Ok(Tokenizer::Bpe(BpeModel::new(0, cap)?));
        }
        let width = add(BASE_VOCABULARY, self.requested_merges - 1)?;
        let cells = width.checked_mul(width).ok_or(ArithmeticOverflow)?;
        let touched_max = self.corpus_bytes.min(cells);
        let mut work = self.limits.training_work;
        charge(&mut work, cells)?;
        check_capacity(
            add(corpus, add(mul::<u64>(cells)?, mul::<usize>(touched_max)?)?)?,
            cap,
        )?;
        let mut counts = allocate(cells)?;
        let mut touched = allocate(touched_max)?;
        let fixed = add(corpus, add(bytes_used(&counts)?, bytes_used(&touched)?)?)?;
        check_capacity(fixed, cap)?;
        counts.resize(cells, 0);
        let available = cap - fixed;
        let mut model = BpeModel::new(self.requested_merges, available)?;
        for rank in 0..self.requested_merges {
            for document in &self.documents {
                charge(&mut work, add(1, document.len().saturating_sub(1))?)?;
                for window in document.windows(2) {
                    let index = window[0] as usize * width + window[1] as usize;
                    if counts[index] == 0 {
                        touched.push(index);
                    }
                    counts[index] = increment(counts[index])?;
                }
            }
            charge(&mut work, touched.len())?;
            let mut best = (1, usize::MAX);
            for index in touched.drain(..) {
                let frequency = std::mem::take(&mut counts[index]);
                if frequency >= 2 && (frequency > best.0 || (frequency == best.0 && index < best.1))
                {
                    best = (frequency, index);
                }
            }
            if best.0 < 2 {
                break;
            }
            let pair = ((best.1 / width) as TokenId, (best.1 % width) as TokenId);
            // Charge expansion verification/copy and the bounded sorted-table insertion.
            charge(
                &mut work,
                add(
                    add(
                        model.expansion(pair.0)?.len(),
                        model.expansion(pair.1)?.len(),
                    )?,
                    add(rank, rank)?,
                )?,
            )?;
            model.append(pair, None, self.limits.expansion_bytes, available)?;
            let id = merge_id(rank)?;
            for document in &mut self.documents {
                charge(&mut work, add(1, document.len())?)?;
                replace(document, pair, id);
            }
        }
        Ok(Tokenizer::Bpe(model))
    }
}

fn increment(value: u64) -> Result<u64> {
    value.checked_add(1).ok_or(ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    #[test]
    fn frequency_overflow() {
        assert_eq!(super::increment(u64::MAX), Err(super::ArithmeticOverflow));
    }
}
