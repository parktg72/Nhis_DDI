#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_원문추출_AI실증보고서.txt 재생성기.

verify_codex.md 는 이 스크립트가 만든 평문 추출본의 행 번호를 인용한다.
추출본 자체는 용량(275KB) 때문에 보관하지 않으므로, 행 번호를 되짚어야 할 때
아래를 실행해 동일한 파일을 다시 만든다. 출력은 결정적이다.

    python3 _원문추출_재생성.py [원본docx경로] > _원문추출_AI실증보고서.txt

주의: 반드시 **무변경 원본**(AI실증보고서_다재약물_DDI_위험예측.docx)을 쓸 것.
개정본은 수식이 OMML 로 바뀌어 행 구성이 달라진다.
"""
import sys
import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

DEFAULT = '/mnt/c/Users/ptg/Downloads/AI실증보고서_다재약물_DDI_위험예측.docx'

def main(path):
    d = docx.Document(path)
    out = []
    for child in d.element.body.iterchildren():
        if child.tag == qn('w:p'):
            p = Paragraph(child, d)
            st = p.style.name if p.style else ''
            t = p.text.strip()
            if t:
                out.append(f"[{st}] {t}" if st and st != 'Normal' else t)
        elif child.tag == qn('w:tbl'):
            tb = Table(child, d)
            out.append("=== TABLE ===")
            for r in tb.rows:
                out.append(" | ".join(c.text.replace('\n', ' / ').strip() for c in r.cells))
            out.append("=== /TABLE ===")
    print("\n".join(out))

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
