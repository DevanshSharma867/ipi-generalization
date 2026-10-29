import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from make_findings import owns


def test_owns_globs_and_exclusions():
    e = {"macros": ["VTwoProbe*", "!*Vsum*"]}
    assert owns(e, "VTwoProbeJudgeQwenTwoFiveSevenbPOneStart")
    assert not owns(e, "VTwoProbeJudgeQwenTwoFiveSevenbVsumPOneStart")
    assert not owns(e, "VTwoTprProtectaiVTwoTestAuroc")
    assert not owns({"macros": ["!*Vsum*"]}, "VTwoProbeTfidfCOnePOneStart")  # exclusions alone own nothing
