from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_85604_model_dataset.sbatch"


def test_launcher_is_rocky9_cpu_only_nonoverwriting_and_syntax_valid():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "PAPER0_EXPECTED_COMMIT",
        "--partition=gen",
        "--qos=gen",
        "--ntasks=8",
        "--cpus-per-task=2",
        "--mem=48G",
        "--no-requeue",
        "SHARD_COUNT=8",
        "--exclusive",
        "--exact",
        "--mem=5G",
        'VERSION_ID%%.*}" != "9"',
        "Refusing to overwrite",
        "status --porcelain --untracked-files=all",
        "model_dataset_manifest.json",
        "normalization.json",
        "artifact_sha256.txt",
        "Training remains closed",
    ):
        assert required in text
    assert "--gres=gpu" not in text
    assert "85606" not in text


def test_launcher_locks_every_internal_file_and_source():
    text = LAUNCHER.read_text(encoding="utf-8")
    for digest in (
        "f60bcb4109c55b2927017017e0a59fc43173fbb8f648525c651f72656c910c97",
        "3d6ffe8d0805e42b46c141a1af2b2fbf5a002cec4ad9c94a980614df0046575d",
        "3312d6ac093e7b9c6ad69a3d7b97fe6aa06695f03c2f10bed7a75f6095795416",
        "12612b2cd65ac807ef4e55996712f6dde49dfca1b956f449ed189386eb5ea04e",
        "3020c5262cc7dab1e14bafbb200a19229159469ceeff2e62def7483f0e9e8d13",
        "f590e6108db5f5f8ac53ddc62bbd96b8576303d675e861c3d38ee8b47235ebc2",
        "843f9ae99d08fbcdabce977b53e4f6b49be05641a82a387d100b237224b77777",
        "a17b536856c6b8108c0553c300200e074e41407129e47ef402a4de51882ea1ba",
        "f4aae5c13ecd944f51cec0c3539f57ff669cb0bb0405cb813c3f50ff6cb83817",
        "eed18a7f7a356a4f8d437647b73d4f8078a5309e1e8583a1089e622196ce4d43",
        "2922c0faadf23a68b43189890f6c0de69dd3cc03442b827c1b8f7e95b8d61e63",
    ):
        assert digest in text


def test_launcher_runs_all_shards_before_the_single_reducer():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "SHARD_PIDS" in text
    assert "At least one model-dataset shard failed" in text
    assert 'if [[ "${#PARTIAL_OUTPUTS[@]}" -ne "${SHARD_COUNT}" ]]' in text
    assert 'if [[ "${#SHARD_OUTPUTS[@]}" -ne "${SHARD_COUNT}" ]]' in text
    assert text.index('"${BUILDER}" \\') < text.index('"${MERGER}" \\')
