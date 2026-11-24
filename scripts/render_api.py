#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright Contributors to the KREPE project

"""Render KREPE's public functions and classes from Doxygen XML as Markdown."""

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def literal(node):
    return "".join(node.itertext()) if node is not None else ""


def text(node):
    return " ".join(literal(node).split())


def slug(name):
    return re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")


def link(refid):
    path = paths[refid]
    relative = "" if path == current_page else path.name
    return f"{relative}#{anchors[refid]}"


def render(node, level=2):
    if node is None:
        return ""
    tag = node.tag
    if tag == "computeroutput":
        value = literal(node).strip()
        delimiter = "``" if "`" in value else "`"
        return f"{delimiter}{value}{delimiter}"
    if tag == "programlisting":
        code = "\n".join(literal(line).rstrip() for line in node.findall("codeline"))
        return f"\n\n```cpp\n{code}\n```\n\n"
    if tag == "linebreak":
        return "  \n"
    if tag == "parameterlist":
        labels = {"param": "Parameters", "templateparam": "Template parameters",
                  "exception": "Exceptions", "retval": "Return values"}
        rows = []
        for item in node.findall("parameteritem"):
            names = ", ".join(text(name) for name in item.findall("parameternamelist/parametername"))
            description = render(item.find("parameterdescription"), level).strip()
            rows.append(f"- `{names}`: " + description.replace("\n", "\n    "))
        return f"\n\n{'#' * level} {labels.get(node.get('kind'), 'Parameters')}\n\n" + "\n".join(rows) + "\n\n"
    if tag in ("itemizedlist", "orderedlist"):
        rows = []
        for index, item in enumerate(node.findall("listitem"), 1):
            marker = "- " if tag == "itemizedlist" else f"{index}. "
            value = render(item, level).strip().replace("\n", "\n" + " " * max(4, len(marker)))
            rows.append(marker + value)
        return "\n\n" + "\n".join(rows) + "\n\n"

    content = (node.text or "") + "".join(
        render(child, level) + (child.tail or "") for child in node
    )
    if tag == "ref":
        refid = node.get("refid")
        return f"[{content}]({link(refid)})" if refid in anchors else content
    if tag == "ulink":
        return f"[{content}]({node.get('url')})"
    if tag == "bold":
        return f"**{content}**"
    if tag == "emphasis":
        return f"*{content}*"
    if tag == "simplesect":
        labels = {"pre": "Preconditions", "post": "Postconditions", "return": "Returns",
                  "note": "Note", "see": "See also", "warning": "Warning"}
        label = labels.get(node.get("kind"), node.get("kind", "Note").title())
        return f"\n\n{'#' * level} {label}\n\n{content.strip()}\n\n"
    if tag == "para":
        return content.strip() + "\n\n"
    return content


def description(node, level=2):
    blocks = (render(node.find(tag), level).strip()
              for tag in ("briefdescription", "detaileddescription"))
    return "\n\n".join(filter(None, blocks))


def usage_guide(content):
    content = content.strip()
    if not content:
        return ""
    return '??? abstract "Usage guide"\n\n' + "\n".join(
        "    " + line if line else "" for line in content.splitlines()
    )


def template_prefix(node):
    templates = []
    for param in node.findall("templateparamlist/param"):
        declaration = " ".join(filter(None, (text(param.find("type")), text(param.find("declname")))))
        default = text(param.find("defval"))
        templates.append(declaration + (f" = {default}" if default else ""))
    return f"template<{', '.join(templates)}>\n" if templates else ""


def signature(node, owner=None):
    declaration = text(node.find("definition"))
    if owner:
        before, _, after = declaration.rpartition(owner + "::")
        declaration = before + after
    declaration += text(node.find("argsstring"))
    initializer = text(node.find("initializer"))
    return template_prefix(node) + declaration + (" " + initializer if initializer else "") + ";"


def function_markdown(overloads):
    first = overloads[0]
    name = first.findtext("qualifiedname") or first.findtext("name")
    result = [f"# `{name}` {{ #{anchors[first.get('id')]} }}"]
    for index, member in enumerate(overloads, 1):
        if len(overloads) > 1:
            result.append(f"## Overload {index} {{ #overload-{index} }}")
        result.append(render(member.find("briefdescription")).strip())
        if len(overloads) == 1:
            result.append("## Interface")
        result.extend((f"```cpp\n{signature(member)}\n```",
                       render(member.find("detaileddescription"),
                              3 if len(overloads) > 1 else 2)))
    return "\n\n".join(filter(None, result))


def type_markdown(compound, members):
    name = compound.findtext("compoundname")
    kind = compound.get("kind")
    declaration = template_prefix(compound) + f"{kind} {name} {{\n"
    if kind == "class":
        declaration += " public:\n"
    declaration += "\n".join("  " + signature(member, name).replace("\n", "\n  ")
                             for member in members) + "\n};"
    result = [f"# `{name}` {{ #{anchors[compound.get('id')]} }}",
              description(compound), "## Interface", f"```cpp\n{declaration}\n```",
              "## Members"]
    for member in members:
        result.extend((f"### `{member.findtext('name')}` {{ #{anchors[member.get('id')]} }}",
                       f"```cpp\n{signature(member, name)}\n```", description(member, 4)))
    return "\n\n".join(filter(None, result))


def main(xml_directory, output):
    global anchors, paths, current_page
    output = Path(output)
    xml_directory = Path(xml_directory)

    def read_compound(refid):
        compound = ET.parse(xml_directory / f"{refid}.xml").find("compounddef")
        for space in compound.iter("sp"):
            space.text = " "
        return compound

    namespace = read_compound("namespacekrepe")
    mainpage = read_compound("indexpage")
    compounds = [namespace]
    for compound in compounds:
        for child in compound.findall("innerclass[@prot='public']"):
            compounds.append(read_compound(child.get("refid")))

    members = {}
    for compound in compounds:
        public = {member.get("id"): member
                  for member in compound.findall("sectiondef/memberdef[@prot='public']")
                  if member.get("kind") in ("function", "variable", "typedef")}
        members[compound.get("id")] = list(public.values())
    functions = {}
    for member in members[namespace.get("id")]:
        functions.setdefault(member.findtext("name"), []).append(member)
    anchors = {}
    counts = {}

    def assign(refid, name):
        base = slug(name)
        number = counts.get(base, 0)
        counts[base] = number + 1
        anchors[refid] = base + (f"-{number}" if number else "")
        return anchors[refid]

    title = mainpage.findtext("title")
    title_anchor = assign(mainpage.get("id"), title)
    assign(namespace.get("id"), namespace.findtext("compoundname").split("::")[-1])

    entries = {}
    for name, overloads in functions.items():
        key = overloads[0].get("id")
        entries[key] = ("function", overloads)
        anchor = assign(key, name)
        for member in overloads:
            anchors[member.get("id")] = anchor
    for compound in compounds[1:]:
        key = compound.get("id")
        entries[key] = ("type", compound)
        assign(key, compound.findtext("compoundname").split("::")[-1])
        for member in members[key]:
            assign(member.get("id"), member.findtext("name"))

    paths = dict.fromkeys(anchors, output)
    for key, (kind, value) in entries.items():
        page = output.with_name(f"{anchors[key]}.md")
        page = page.with_suffix(".symbol.md") if page == output else page
        paths[key] = page
        page_members = value if kind == "function" else members[key]
        paths.update((member.get("id"), page) for member in page_members)

    def overview(content):
        rows = ["| Name | Description |", "| --- | --- |"]
        for key in content:
            kind, value = entries[key]
            node = value[0] if kind == "function" else value
            name = node.findtext("name") or node.findtext("compoundname").split("::")[-1]
            summary = " ".join(render(node.find("briefdescription")).split()).replace("|", "\\|")
            if kind == "function" and len(value) > 1:
                summary += f" ({len(value)} overloads)"
            rows.append(f'| <a id="{anchors[key]}"></a>[`{name}`]({link(key)}) | {summary} |')
        return "\n".join(rows)

    def write(path, blocks):
        body = "\n\n".join(filter(None, map(str.strip, blocks)))
        path.write_text(body + "\n", encoding="utf-8")

    current_page = output
    introduction, _, guide = description(mainpage).partition("\n\n")
    blocks = [f"# {title} {{ #{title_anchor} }}", introduction,
              f'<a id="{anchors[namespace.get("id")]}"></a>', description(namespace)]
    for kind, heading in (("function", "Functions"), ("type", "Types")):
        content = [key for key, (entry_kind, _) in entries.items() if entry_kind == kind]
        if content:
            blocks.extend((f"## {heading}", overview(content)))
    blocks.append(usage_guide(guide))
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, blocks)
    for key, (kind, value) in entries.items():
        current_page = paths[key]
        content = function_markdown(value) if kind == "function" else type_markdown(value, members[key])
        write(current_page, [f"[{title}]({output.name})", content])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_directory")
    parser.add_argument("output")
    main(**vars(parser.parse_args()))
