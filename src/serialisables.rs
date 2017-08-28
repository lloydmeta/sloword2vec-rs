use serde::ser::{Serialize, Serializer, SerializeStruct};
use serde::de;
use serde::de::{Deserialize, Deserializer, Visitor, SeqAccess, MapAccess, Unexpected};
use std::fmt;

use ndarray::Array2;

// Newtype so we can define serialisation for Matrix
#[derive(PartialEq, Debug)]
pub(crate) struct SerializableMatrix(pub(crate) Array2<f32>);

impl Serialize for SerializableMatrix {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SerializableMatrix", 3)?;
        state.serialize_field("rows", &self.0.rows())?;
        state.serialize_field("cols", &self.0.cols())?;
        let as_vec: Vec<_> = self.0.iter().collect();
        state.serialize_field("data", &as_vec)?;
        state.end()
    }
}


impl<'de> Deserialize<'de> for SerializableMatrix {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Rows,
            Cols,
            Data,
        };

        struct SerializableMatrixVisitor;

        impl<'de> Visitor<'de> for SerializableMatrixVisitor {
            type Value = SerializableMatrix;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct SerializableMatrix")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<SerializableMatrix, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let rows = seq.next_element()?.ok_or_else(
                    || de::Error::invalid_length(0, &self),
                )?;
                let cols = seq.next_element()?.ok_or_else(
                    || de::Error::invalid_length(1, &self),
                )?;
                let data: Vec<f32> = seq.next_element()?.ok_or_else(
                    || de::Error::invalid_length(2, &self),
                )?;
                let underlying = Array2::from_shape_vec((rows, cols), data).map_err(|_| {
                    de::Error::invalid_value(
                        Unexpected::Other("Data that does not match dimensions"),
                        &self,
                    )
                })?;
                Ok(SerializableMatrix(underlying))
            }

            fn visit_map<V>(self, mut map: V) -> Result<SerializableMatrix, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut rows = None;
                let mut cols = None;
                let mut data = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Rows => {
                            if rows.is_some() {
                                return Err(de::Error::duplicate_field("rows"));
                            }
                            rows = Some(map.next_value()?);
                        }
                        Field::Cols => {
                            if cols.is_some() {
                                return Err(de::Error::duplicate_field("cols"));
                            }
                            cols = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                    }
                }
                let rows = rows.ok_or_else(|| de::Error::missing_field("rows"))?;
                let cols = cols.ok_or_else(|| de::Error::missing_field("cols"))?;
                let data: Vec<f32> = data.ok_or_else(|| de::Error::missing_field("data"))?;
                let underlying = Array2::from_shape_vec((rows, cols), data).map_err(|_| {
                    de::Error::invalid_value(
                        Unexpected::Other("Data that does not match dimensions"),
                        &self,
                    )
                })?;
                Ok(SerializableMatrix(underlying))
            }
        }
        const FIELDS: &'static [&'static str] = &["rows", "cols", "data"];
        deserializer.deserialize_struct(
            "SerializableMatrixVisitor",
            FIELDS,
            SerializableMatrixVisitor,
        )
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use serde_json;

    #[test]
    fn test_roundtrip() {
        let s_mat = SerializableMatrix(Array2::eye(3));
        let as_json = serde_json::to_string(&s_mat).expect("to json to work");
        let deserialised = serde_json::from_str(&as_json).expect("from json to work");
        assert_eq!(s_mat, deserialised);
    }
}
