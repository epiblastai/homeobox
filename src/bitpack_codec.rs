//! zarrs v3 BytesToBytesCodec implementation for bitpacking.
//!
//! Registered at runtime via `register_codec_v3()` so that `CodecChain::from_metadata`
//! picks it up automatically — zero changes to `RustBatchReader`.

use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use zarrs::array::{
    ArrayBytesRaw, BytesToBytesCodecTraits, BytesRepresentation, Codec, CodecError,
    CodecMetadataOptions, CodecOptions, CodecTraits, RecommendedConcurrency,
};
use zarrs::array::codec::api::{
    CodecRuntimePluginV3, CodecRuntimeRegistryHandleV3,
    PartialDecoderCapability, PartialEncoderCapability,
    register_codec_v3,
};
use zarrs::metadata::{Configuration, ConfigurationSerialize};
use zarrs::plugin::{ExtensionName, ZarrVersion};

use crate::bitpacking::{self, Transform};

/// Codec name used in zarr v3 metadata.
const CODEC_NAME: &str = "homeobox.bitpacking";

// ---------------------------------------------------------------------------
// Configuration (serialized into zarr array JSON)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BitpackCodecConfiguration {
    pub transform: String,
    pub element_size: usize,
}

impl ConfigurationSerialize for BitpackCodecConfiguration {}

// ---------------------------------------------------------------------------
// Codec struct
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BitpackCodec {
    transform: Transform,
    element_size: usize,
}

impl BitpackCodec {
    pub fn new(transform: Transform, element_size: usize) -> Self {
        Self { transform, element_size }
    }

    pub fn from_configuration(config: &BitpackCodecConfiguration) -> Result<Self, String> {
        let transform = Transform::from_str(&config.transform)?;
        if config.element_size != 4 {
            return Err(format!(
                "bitpacking codec only supports element_size=4 (uint32), got {}",
                config.element_size
            ));
        }
        Ok(Self::new(transform, config.element_size))
    }
}

// ---------------------------------------------------------------------------
// ExtensionName
// ---------------------------------------------------------------------------

impl ExtensionName for BitpackCodec {
    fn name(&self, version: ZarrVersion) -> Option<Cow<'static, str>> {
        match version {
            ZarrVersion::V3 => Some(Cow::Borrowed(CODEC_NAME)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CodecTraits
// ---------------------------------------------------------------------------

impl CodecTraits for BitpackCodec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn configuration(
        &self,
        version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        match version {
            ZarrVersion::V3 => {
                let config = BitpackCodecConfiguration {
                    transform: self.transform.as_str().to_string(),
                    element_size: self.element_size,
                };
                Some(Configuration::from(config))
            }
            _ => None,
        }
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

// ---------------------------------------------------------------------------
// BytesToBytesCodecTraits
// ---------------------------------------------------------------------------

impl BytesToBytesCodecTraits for BitpackCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        decoded_value: ArrayBytesRaw<'a>,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let bytes = decoded_value.as_ref();
        if bytes.len() % self.element_size != 0 {
            return Err(CodecError::Other(
                format!(
                    "bitpacking: input length {} is not a multiple of element_size {}",
                    bytes.len(),
                    self.element_size
                )
                .into(),
            ));
        }

        // Reinterpret as u32 (native endian — zarr bytes codec handles endianness)
        let values: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();

        let encoded = bitpacking::encode(&values, self.transform);
        Ok(Cow::Owned(encoded))
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let values = bitpacking::decode(encoded_value.as_ref())
            .map_err(|e| CodecError::Other(e.into()))?;

        // Convert u32 values back to bytes (native endian)
        let mut out = Vec::with_capacity(values.len() * 4);
        for v in &values {
            out.extend_from_slice(&v.to_ne_bytes());
        }
        Ok(Cow::Owned(out))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        // Bitpacking output size is variable (depends on data), so report bounded
        decoded_representation.size().map_or(
            BytesRepresentation::UnboundedSize,
            |size| {
                // Worst case: header (5) + ceil(N/128) * (1 + 128*4) per block
                // But generally much smaller. Report bounded at input size + overhead.
                let num_elements = size / self.element_size as u64;
                let num_blocks = num_elements.div_ceil(128);
                let max_encoded = 5 + num_blocks * (1 + 128 * 4);
                BytesRepresentation::BoundedSize(max_encoded)
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Runtime registration
// ---------------------------------------------------------------------------

/// Register the bitpacking codec with zarrs' v3 codec registry.
/// Returns a handle that must be kept alive for the codec to remain registered.
pub fn register_bitpack_codec() -> CodecRuntimeRegistryHandleV3 {
    register_codec_v3(CodecRuntimePluginV3::new(
        |name| name == CODEC_NAME,
        |metadata| {
            let config: BitpackCodecConfiguration = metadata
                .to_typed_configuration()
                .map_err(|e| zarrs::plugin::PluginCreateError::Other(e.to_string().into()))?;
            let codec = BitpackCodec::from_configuration(&config)
                .map_err(|e| zarrs::plugin::PluginCreateError::Other(e.into()))?;
            Ok(Codec::BytesToBytes(Arc::new(codec)))
        },
    ))
}
