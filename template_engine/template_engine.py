from pathlib import Path
from itertools import product
import json
import sys


BEGIN_TEMPLATE = "//! begin template"
END_TEMPLATE = "//! end template"


with open(sys.argv[1]) as f:
    config = json.load(f)

file_patterns = config["file_patterns"]
keywords = config["keywords"]

root = Path(sys.argv[2])


def generate_output(line_iter, file_path):
    yield "// ***********************************************************\n"
    yield "// *       This is an automatically generated file.          *\n"
    yield "// *       For editing, please use the source file:          *\n"
    yield "// " + file_path.name + "\n"
    yield "// ***********************************************************\n\n"

    while True:
        try:
            line = next(line_iter)
        except StopIteration:
            return

        if BEGIN_TEMPLATE in line:
            template = list(read_template(line_iter))
            yield from expand_template(template)
        else:
            yield line


def read_template(line_iter):
    while True:
        line = next(line_iter)
        if END_TEMPLATE in line:
            return
        yield line


def expand_template(template):
    matching_keywords = {
        keyword: instances for keyword, instances in keywords.items()
        if any(keyword in line for line in template)
    }

    for instances in product(*matching_keywords.values()):
        # use dict as a proxy for an ordered set
        flags = {}
        for instance in instances:
            if instance["flag"] is None:
                continue

            if isinstance(instance["flag"], (list, tuple)):
                for flag in instance["flag"]:
                    flags[flag] = None
            else:
                flags[instance["flag"]] = None

        if flags:
            yield r"#if " + " && ".join(f"defined({flag})" for flag in flags) + "\n"

        for line in template:
            for keyword, instance in zip(matching_keywords, instances):
                modifiers = {name: code for name, code in instance.items() if name not in ("flag", "code")}

                for i, name in enumerate(modifiers):
                    line = line.replace(keyword + f"{{name}}", f"#{i}")

                line = line.replace(keyword, instance["code"])

                for i, code in enumerate(modifiers.values()):
                    line = line.replace(f"#{i}", code)

            yield line

        if flags:
            yield "#endif\n"


for file_pattern in file_patterns:
    for file_path in root.glob(file_pattern):
        print(file_path)

        with open(file_path) as f:
            lines = f.readlines()

        output = list(generate_output(iter(lines), file_path))

        with open(str(file_path)[:-len(file_path.suffix)], 'w') as f:
            f.writelines(output)
