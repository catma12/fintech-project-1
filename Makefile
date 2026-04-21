.PHONY: all shocks features train shap report clean

all: shocks features train shap report

shocks:
	python -m src.var_shocks

features:
	python -m src.features

targets:
	python -m src.targets

train:
	python -m src.train

shap:
	python -m src.shap_analysis

report:
	python -m src.report

clean:
	rm -rf outputs/shocks outputs/features outputs/targets outputs/models outputs/oof outputs/shap outputs/report outputs/run_manifest.json
