use super::ShaderDefVal;
use crate::define_atomic_id;
use bevy_asset::{io::Reader, uuid::Uuid, Asset, AssetLoader, AssetPath, Handle, LoadContext};
use bevy_reflect::TypePath;
use bevy_utils::tracing::error;
use futures_lite::AsyncReadExt;
use std::{borrow::Cow, marker::Copy};
use thiserror::Error;

define_atomic_id!(ShaderId);

#[derive(Error, Debug)]
pub enum ShaderReflectError {
    #[error(transparent)]
    WgslParse(#[from] naga::front::wgsl::ParseError),
    #[cfg(feature = "shader_format_glsl")]
    #[error("GLSL Parse Error: {0:?}")]
    GlslParse(Vec<naga::front::glsl::Error>),
    #[cfg(feature = "shader_format_spirv")]
    #[error(transparent)]
    SpirVParse(#[from] naga::front::spv::Error),
    #[error(transparent)]
    Validation(#[from] naga::WithSpan<naga::valid::ValidationError>),
}
/// A shader, as defined by its [`ShaderSource`](wgpu::ShaderSource) and [`ShaderStage`](naga::ShaderStage)
/// This is an "unprocessed" shader. It can contain preprocessor directives.
#[derive(Asset, TypePath, Debug, Clone)]
pub struct Shader {
    pub path: String,
    pub source: Source,
    pub import_path: ShaderImport,
    pub imports: Vec<ShaderImport>,
    // extra imports not specified in the source string
    pub additional_imports: Vec<naga_oil::compose::ImportDefinition>,
    // any shader defs that will be included when this module is used
    pub shader_defs: Vec<ShaderDefVal>,
    // we must store strong handles to our dependencies to stop them
    // from being immediately dropped if we are the only user.
    pub file_dependencies: Vec<Handle<Shader>>,
}

impl Shader {
    fn preprocess(source: &str, path: &str) -> (ShaderImport, Vec<ShaderImport>) {
        let (import_path, imports, _) = naga_oil::compose::get_preprocessor_data(source);

        let import_path = import_path
            .map(ShaderImport::Custom)
            .unwrap_or_else(|| ShaderImport::AssetPath(path.to_owned()));

        let imports = imports
            .into_iter()
            .map(|import| {
                if import.import.starts_with('\"') {
                    let import = import
                        .import
                        .chars()
                        .skip(1)
                        .take_while(|c| *c != '\"')
                        .collect();
                    ShaderImport::AssetPath(import)
                } else {
                    ShaderImport::Custom(import.import)
                }
            })
            .collect();

        (import_path, imports)
    }

    pub fn from_wgsl(source: impl Into<Cow<'static, str>>, path: impl Into<String>) -> Shader {
        let source = source.into();
        let path = path.into();
        let (import_path, imports) = Shader::preprocess(&source, &path);
        Shader {
            path,
            imports,
            import_path,
            source: Source::Wgsl(source),
            additional_imports: Default::default(),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
        }
    }

    pub fn from_wgsl_with_defs(
        source: impl Into<Cow<'static, str>>,
        path: impl Into<String>,
        shader_defs: Vec<ShaderDefVal>,
    ) -> Shader {
        Self {
            shader_defs,
            ..Self::from_wgsl(source, path)
        }
    }

    pub fn from_glsl(
        source: impl Into<Cow<'static, str>>,
        stage: naga::ShaderStage,
        path: impl Into<String>,
    ) -> Shader {
        let source = source.into();
        let path = path.into();
        let (import_path, imports) = Shader::preprocess(&source, &path);
        Shader {
            path,
            imports,
            import_path,
            source: Source::Glsl(source, stage),
            additional_imports: Default::default(),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
        }
    }

    pub fn from_spirv(source: impl Into<Cow<'static, [u8]>>, path: impl Into<String>) -> Shader {
        let path = path.into();
        Shader {
            path: path.clone(),
            imports: Vec::new(),
            import_path: ShaderImport::AssetPath(path),
            source: Source::SpirV(source.into()),
            additional_imports: Default::default(),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
        }
    }

    pub fn set_import_path<P: Into<String>>(&mut self, import_path: P) {
        self.import_path = ShaderImport::Custom(import_path.into());
    }

    #[must_use]
    pub fn with_import_path<P: Into<String>>(mut self, import_path: P) -> Self {
        self.set_import_path(import_path);
        self
    }

    #[inline]
    pub fn import_path(&self) -> &ShaderImport {
        &self.import_path
    }

    pub fn imports(&self) -> impl ExactSizeIterator<Item = &ShaderImport> {
        self.imports.iter()
    }
}

impl<'a> From<&'a Shader> for naga_oil::compose::ComposableModuleDescriptor<'a> {
    fn from(shader: &'a Shader) -> Self {
        let shader_defs = shader
            .shader_defs
            .iter()
            .map(|def| match def {
                ShaderDefVal::Bool(name, b) => {
                    (name.clone(), naga_oil::compose::ShaderDefValue::Bool(*b))
                }
                ShaderDefVal::Int(name, i) => {
                    (name.clone(), naga_oil::compose::ShaderDefValue::Int(*i))
                }
                ShaderDefVal::UInt(name, i) => {
                    (name.clone(), naga_oil::compose::ShaderDefValue::UInt(*i))
                }
            })
            .collect();

        let as_name = match &shader.import_path {
            ShaderImport::AssetPath(asset_path) => Some(format!("\"{asset_path}\"")),
            ShaderImport::Custom(_) => None,
        };

        naga_oil::compose::ComposableModuleDescriptor {
            source: shader.source.as_str(),
            file_path: &shader.path,
            language: (&shader.source).into(),
            additional_imports: &shader.additional_imports,
            shader_defs,
            as_name,
        }
    }
}

impl<'a> From<&'a Shader> for naga_oil::compose::NagaModuleDescriptor<'a> {
    fn from(shader: &'a Shader) -> Self {
        naga_oil::compose::NagaModuleDescriptor {
            source: shader.source.as_str(),
            file_path: &shader.path,
            shader_type: (&shader.source).into(),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone)]
pub enum Source {
    Wgsl(Cow<'static, str>),
    Glsl(Cow<'static, str>, naga::ShaderStage),
    SpirV(Cow<'static, [u8]>),
    // TODO: consider the following
    // PrecompiledSpirVMacros(HashMap<HashSet<String>, Vec<u32>>)
    // NagaModule(Module) ... Module impls Serialize/Deserialize
}

impl Source {
    pub fn as_str(&self) -> &str {
        match self {
            Source::Wgsl(s) | Source::Glsl(s, _) => s,
            Source::SpirV(_) => panic!("spirv not yet implemented"),
        }
    }
}

impl From<&Source> for naga_oil::compose::ShaderLanguage {
    fn from(value: &Source) -> Self {
        match value {
            Source::Wgsl(_) => naga_oil::compose::ShaderLanguage::Wgsl,
            #[cfg(any(feature = "shader_format_glsl", target_arch = "wasm32"))]
            Source::Glsl(_, _) => naga_oil::compose::ShaderLanguage::Glsl,
            #[cfg(all(not(feature = "shader_format_glsl"), not(target_arch = "wasm32")))]
            Source::Glsl(_, _) => panic!(
                "GLSL is not supported in this configuration; use the feature `shader_format_glsl`"
            ),
            Source::SpirV(_) => panic!("spirv not yet implemented"),
        }
    }
}

impl From<&Source> for naga_oil::compose::ShaderType {
    fn from(value: &Source) -> Self {
        match value {
            Source::Wgsl(_) => naga_oil::compose::ShaderType::Wgsl,
            #[cfg(any(feature = "shader_format_glsl", target_arch = "wasm32"))]
            Source::Glsl(_, shader_stage) => match shader_stage {
                naga::ShaderStage::Vertex => naga_oil::compose::ShaderType::GlslVertex,
                naga::ShaderStage::Fragment => naga_oil::compose::ShaderType::GlslFragment,
                naga::ShaderStage::Compute => panic!("glsl compute not yet implemented"),
            },
            #[cfg(all(not(feature = "shader_format_glsl"), not(target_arch = "wasm32")))]
            Source::Glsl(_, _) => panic!(
                "GLSL is not supported in this configuration; use the feature `shader_format_glsl`"
            ),
            Source::SpirV(_) => panic!("spirv not yet implemented"),
        }
    }
}

#[derive(Default)]
pub struct ShaderLoader;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ShaderLoaderError {
    #[error("Could not load shader: {0}")]
    Io(#[from] std::io::Error),
    #[error("Could not parse shader: {0}")]
    Parse(#[from] std::string::FromUtf8Error),
}

impl AssetLoader for ShaderLoader {
    type Asset = Shader;
    type Settings = ();
    type Error = ShaderLoaderError;
    async fn load<'a>(
        &'a self,
        reader: &'a mut Reader<'_>,
        _settings: &'a Self::Settings,
        load_context: &'a mut LoadContext<'_>,
    ) -> Result<Shader, Self::Error> {
        let ext = load_context.path().extension().unwrap().to_str().unwrap();
        let path = load_context.asset_path().to_string();
        // On windows, the path will inconsistently use \ or /.
        // TODO: remove this once AssetPath forces cross-platform "slash" consistency. See #10511
        let path = path.replace(std::path::MAIN_SEPARATOR, "/");
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let mut shader = match ext {
            "spv" => Shader::from_spirv(bytes, load_context.path().to_string_lossy()),
            "wgsl" => Shader::from_wgsl(String::from_utf8(bytes)?, path),
            "vert" => Shader::from_glsl(String::from_utf8(bytes)?, naga::ShaderStage::Vertex, path),
            "frag" => {
                Shader::from_glsl(String::from_utf8(bytes)?, naga::ShaderStage::Fragment, path)
            }
            "comp" => {
                Shader::from_glsl(String::from_utf8(bytes)?, naga::ShaderStage::Compute, path)
            }
            _ => panic!("unhandled extension: {ext}"),
        };

        // collect and store file dependencies
        for import in &shader.imports {
            if let ShaderImport::AssetPath(asset_path) = import {
                shader.file_dependencies.push(load_context.load(asset_path));
            }
        }
        Ok(shader)
    }

    fn extensions(&self) -> &[&str] {
        &["spv", "wgsl", "vert", "frag", "comp"]
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ShaderImport {
    AssetPath(String),
    Custom(String),
}

impl ShaderImport {
    pub fn module_name(&self) -> Cow<'_, String> {
        match self {
            ShaderImport::AssetPath(s) => Cow::Owned(format!("\"{s}\"")),
            ShaderImport::Custom(s) => Cow::Borrowed(s),
        }
    }
}

/// A reference to a shader asset.
pub enum ShaderRef {
    /// Use the "default" shader for the current context.
    Default,
    /// A handle to a shader stored in the [`Assets<Shader>`](bevy_asset::Assets) resource
    Handle(Handle<Shader>),
    /// An asset path leading to a shader
    Path(AssetPath<'static>),
    /// A constant Uuid reference to a shader handle in the InternalAssets resource
    InternalAsset(Uuid),
}

impl From<Handle<Shader>> for ShaderRef {
    fn from(handle: Handle<Shader>) -> Self {
        Self::Handle(handle)
    }
}

impl From<AssetPath<'static>> for ShaderRef {
    fn from(path: AssetPath<'static>) -> Self {
        Self::Path(path)
    }
}

impl From<&'static str> for ShaderRef {
    fn from(path: &'static str) -> Self {
        Self::Path(AssetPath::from(path))
    }
}

impl From<Uuid> for ShaderRef {
    fn from(uuid: Uuid) -> Self {
        Self::InternalAsset(uuid)
    }
}
