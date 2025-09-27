class Sam2PredictorBuilder:
    def build(self, constructors, resolved_configuration):
        build_sam2_constructor, Sam2ImagePredictorClass = constructors
        sam2_model = build_sam2_constructor(
            str(resolved_configuration["model_config_name"]),
            str(resolved_configuration["checkpoint_path"]),
        )
        image_predictor = Sam2ImagePredictorClass(sam2_model)
        device_name = resolved_configuration["device"]
        image_predictor.model.to(device_name)
        return image_predictor
