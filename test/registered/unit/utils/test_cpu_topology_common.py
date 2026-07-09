import unittest
from unittest.mock import patch

from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ProcessWithoutAffinity:
    pass


class TestCpuTopologyFallbacks(unittest.TestCase):
    @patch("sglang.srt.utils.common.psutil.cpu_count", return_value=4)
    @patch(
        "sglang.srt.utils.common.subprocess.check_output",
        side_effect=FileNotFoundError("lscpu"),
    )
    def test_parse_lscpu_topology_falls_back_to_single_node(
        self, _mock_check_output, _mock_cpu_count
    ):
        self.assertEqual(
            common.parse_lscpu_topology(),
            [(0, 0, 0, 0), (1, 1, 0, 0), (2, 2, 0, 0), (3, 3, 0, 0)],
        )

    @patch("sglang.srt.utils.common.psutil.cpu_count", return_value=4)
    @patch(
        "sglang.srt.utils.common.parse_lscpu_topology",
        return_value=[(0, 0, 0, 0), (1, 1, 0, 0)],
    )
    @patch(
        "sglang.srt.utils.common.psutil.Process",
        return_value=_ProcessWithoutAffinity(),
    )
    def test_get_physical_cpus_by_numa_handles_missing_cpu_affinity(
        self, _mock_process, _mock_parse, _mock_cpu_count
    ):
        self.assertEqual(common.get_physical_cpus_by_numa(), {0: {0, 1}})


if __name__ == "__main__":
    unittest.main()
