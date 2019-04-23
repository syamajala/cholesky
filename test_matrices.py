"""
Copyright 2019 Stanford University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import subprocess
import os
import shutil
from verify import check_matrix, check_solution


def run(**kwargs):
    args = ["regent.py", "mmat.rg",
            "-i", kwargs["mat"], "-s", kwargs["separators"], "-c", kwargs["clusters"],
            "-b", kwargs["b"], "-o", kwargs["solution"], "-m", kwargs["factored_mat"],
            "-fflow", "0", "-ll:cpu", "3"]
    print(" ".join(args))
    with open(kwargs["stdout"], 'w') as f:
        subprocess.call(args, stdout=f)


def generate_args(args, input_prefix, output_prefix):
    new_args = {"mat": os.path.join(input_prefix, args["mat"]),
                "b": os.path.join(input_prefix, args["b"]),
                "separators": os.path.join(input_prefix, args["separators"]),
                "clusters": os.path.join(input_prefix, args["clusters"]),
                "factored_mat": os.path.join(output_prefix, args["factored_mat"]),
                "solution": os.path.join(output_prefix, args["solution"]),
                "stdout": os.path.join(output_prefix, "stdout")}
    return new_args


class TestMatrices():

    def test_9x9(self):

        input_prefix = "tests/lapl_9x9"
        output_prefix = os.path.join(input_prefix, "output")

        if os.path.exists(output_prefix):
            shutil.rmtree(output_prefix)

        os.makedirs(output_prefix)

        args = {"mat": "lapl_3_2.mtx",
                "b": "B_9x1.mtx",
                "separators": "lapl_3_2_ord_2.txt",
                "clusters": "lapl_3_2_clust_2.txt",
                "factored_mat": "factored_3_2.mtx",
                "solution": "solution_9x1.mtx"}
        args = generate_args(args, input_prefix, output_prefix)

        run(**args)
        assert check_matrix(args['mat'], args['separators'], args['factored_mat']) is True
        assert check_solution(args['mat'], args['b'], args['solution']) is True

    def test_25x25(self):

        input_prefix = "tests/lapl_25x25"
        output_prefix = os.path.join(input_prefix, "output")

        if os.path.exists(output_prefix):
            shutil.rmtree(output_prefix)

        os.makedirs(output_prefix)

        args = {"mat": "lapl_5_2.mtx",
                "b": "B_25x1.mtx",
                "separators": "lapl_5_2_ord_3.txt",
                "clusters": "lapl_5_2_clust_3.txt",
                "factored_mat": "factored_5_2.mtx",
                "solution": "solution_25x1.mtx"}
        args = generate_args(args, input_prefix, output_prefix)

        run(**args)
        assert check_matrix(args['mat'], args['separators'], args['factored_mat']) is True
        assert check_solution(args['mat'], args['b'], args['solution']) is True

    def test_400x400(self):

        input_prefix = "tests/lapl_400x400"
        output_prefix = os.path.join(input_prefix, "output")

        if os.path.exists(output_prefix):
            shutil.rmtree(output_prefix)

        os.makedirs(output_prefix)

        args = {"mat": "lapl_20_2.mtx",
                "b": "B_400x1.mtx",
                "separators": "lapl_20_2_ord_5.txt",
                "clusters": "lapl_20_2_clust_5.txt",
                "factored_mat": "factored_20_2.mtx",
                "solution": "solution_400x1.mtx"}
        args = generate_args(args, input_prefix, output_prefix)

        run(**args)
        assert check_matrix(args['mat'], args['separators'], args['factored_mat']) is True
        assert check_solution(args['mat'], args['b'], args['solution']) is True

    def test_3375x3375(self):

        input_prefix = "tests/lapl_3375x3375"
        output_prefix = os.path.join(input_prefix, "output")

        if os.path.exists(output_prefix):
            shutil.rmtree(output_prefix)

        os.makedirs(output_prefix)

        args = {"mat": "lapl_15_3.mtx",
                "b": "B_3375x1.mtx",
                "separators": "lapl_15_3_ord_5.txt",
                "clusters": "lapl_15_3_clust_5.txt",
                "factored_mat": "factored_15_3.mtx",
                "solution": "solution_3375x1.mtx"}
        args = generate_args(args, input_prefix, output_prefix)

        run(**args)
        assert check_matrix(args['mat'], args['separators'], args['factored_mat']) is True
        assert check_solution(args['mat'], args['b'], args['solution']) is True
