import subprocess
import os
import shutil
from verify import check_matrix, check_solution


def run(**kwargs):
    args = ["regent.py", "mmat.rg",
            "-i", kwargs["mat"], "-s", kwargs["separators"], "-c", kwargs["clusters"],
            "-b", kwargs["b"], "-o", kwargs["solution"], "-p", kwargs["permuted_mat"], "-m", kwargs["factored_mat"],
            "-fflow", "0", "-ll:cpu", "4"]
    print(" ".join(args))
    with open(kwargs["stdout"], 'w') as f:
        subprocess.call(args, stdout=f)


def generate_args(args, input_prefix, output_prefix):
    new_args = {"mat": os.path.join(input_prefix, args["mat"]),
                "b": os.path.join(input_prefix, args["b"]),
                "separators": os.path.join(input_prefix, args["separators"]),
                "clusters": os.path.join(input_prefix, args["clusters"]),
                "permuted_mat": os.path.join(output_prefix, args["permuted_mat"]),
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
                "permuted_mat": "permuted_3_2.mtx",
                "factored_mat": "factored_3_2.mtx",
                "solution": "solution_9x1.mtx"}
        args = generate_args(args, input_prefix, output_prefix)

        run(**args)
        assert check_matrix(args['permuted_mat'], args['factored_mat']) == True
        assert check_solution(args['mat'], args['b'], args['solution']) == True

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
                "permuted_mat": "permuted_5_2.mtx",
                "factored_mat": "factored_5_2.mtx",
                "solution": "solution_25x1.mtx"}
        args = generate_args(args, input_prefix, output_prefix)

        run(**args)
        assert check_matrix(args['permuted_mat'], args['factored_mat']) == True
        assert check_solution(args['mat'], args['b'], args['solution']) == True

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
                "permuted_mat": "permuted_20_2.mtx",
                "factored_mat": "factored_20_2.mtx",
                "solution": "solution_400x1.mtx"}
        args = generate_args(args, input_prefix, output_prefix)

        run(**args)
        assert check_matrix(args['permuted_mat'], args['factored_mat']) == True
        assert check_solution(args['mat'], args['b'], args['solution']) == True
