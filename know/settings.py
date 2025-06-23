class EmbeddingSettings:
    def __init__(
        self,
        *,
        calculator_type: str = "local",          # factory selector
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
        device: str | None = None,
        batch_size: int = 32,
        quantize: bool = False,
        quantize_bits: int = 8,
        enabled: bool = True                    # quick on/off switch
    ):
        self.calculator_type   = calculator_type
        self.model_name        = model_name
        self.normalize_embeddings = normalize_embeddings
        self.device            = device
        self.batch_size        = batch_size
        self.quantize          = quantize
        self.quantize_bits     = quantize_bits
        self.enabled           = enabled


class ProjectSettings:
    project_path: str = None
    project_id: str = None

    def __init__(self
                 , project_path: str = None
                 , project_id: str = None
                 , embedding: EmbeddingSettings = None):
        self.project_path = project_path
        self.project_id = project_id
        self.embedding  = embedding          # <- add this line
