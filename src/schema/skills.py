from __future__ import annotations
import os
import re
import subprocess


class Skill:
    """A skill to be used by the harness."""

    name: str
    description: str
    content: str

    def __init__(self, name: str, description: str, content: str):
        self.name = name
        self.description = description
        self.content = content
    
    def load_skill_body(skill_dir):
        """当任务匹配 description 时，读取完整 SKILL.md 正文"""
        path = os.path.join(skill_dir, "SKILL.md")
        with open(path) as f:
            content = f.read()
        # 去掉 frontmatter，只返回正文
        return re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)

    # ---------- Level 3: 按需访问附属资源 ----------
    def read_reference(skill_dir, rel_path):
        """Agent 通过 Read 类工具读取 reference"""
        full = os.path.join(skill_dir, rel_path)
        with open(full) as f:
            return f.read()

    def run_script(skill_dir, rel_script, args=None):
        """Agent 通过 Bash/Shell 工具执行脚本，解决 CWD 陷阱"""
        script_path = os.path.join(skill_dir, rel_script)
        # 关键：在 Skill 目录下执行，或显式传递绝对路径
        result = subprocess.run(
            ["bash", script_path] + (args or []),
            cwd=skill_dir,               # 让脚本的相对路径依赖正确解析
            capture_output=True, text=True, timeout=30
        )
        return result.stdout or result.stderr
