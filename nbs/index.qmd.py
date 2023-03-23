"""---
title: Home
pagetitle: dabest
page-layout: custom
section-divs: false
css: index.css
toc: false
---"""

from fastcore.foundation import L
from nbdev import qmd

def img(fname, classes=None, **kwargs): return qmd.img(f"images/{fname}", classes=classes, **kwargs)
def btn(txt, link): return qmd.btn(txt, link=link, classes=['btn-action-primary', 'btn-action', 'btn', 'btn-success', 'btn-lg'])
def banner(txt, classes=None, style=None): return qmd.div(txt, L('hero-banner')+classes, style=style)

features = L(
    ('docs', 'Beautiful technical documentation and scientific articles with Quarto'),
    ('testing', 'Out-of-the-box continuous integration with GitHub Actions'),
    ('packaging', 'Publish code to PyPI and conda, and prose to GitHub Pages'),
    ('visualization', 'Estimation plots are robust, beautiful, and convey important statistical information elegantly and efficiently.'),
    ('jupyter', 'Write prose, code, and tests in notebooks'),
    ('git', 'Git-friendly notebooks: human-readable merge conflicts')
)

def industry(im, **kwargs): return qmd.div(img(im, **kwargs), ["g-col-12", "g-col-sm-6", "g-col-md-3"])

def testm(im, nm, detl, txt):
    return qmd.div(f"""{img(im, link=True)}

# {nm}

## {detl}

### {txt}""", ["testimonial", "g-col-12", "g-col-md-6"])


def feature(im, desc): return qmd.div(f"{img(im+'.svg')}\n\n{desc}\n", ['feature', 'g-col-12', 'g-col-sm-6', 'g-col-md-4'])

feature_d = qmd.div('\n'.join(features.starmap(feature)), ['grid', 'gap-4'], style={"padding-bottom": "60px"})

def b(*args, **kwargs): print(banner (*args, **kwargs))
def d(*args, **kwargs): print(qmd.div(*args, **kwargs))

###
# Output section
###

b(f"""# <span style='color:#009AF1'>Data Analysis using</span><br>Bootstrap\-Coupled ESTimation

### Analyze your data with effect sizes and beautiful estimation plots.

{btn('Get started', '/01-getting-started.ipynb')}

{img('showpiece.png', style={"margin-top": "20px", "margin-bottom": "20px"}, link=True)}""", "content-block")

feature_h = banner(f"""## <span style='color:#009AF1'>Robust and Beautiful</span><br>Statistical Visualization

### Estimation statistics is a simple framework that avoids the pitfalls of significance testing. It uses familiar statistical concepts: means, mean differences, and error bars. More importantly, it focuses on the effect size of one's experiment/intervention, as opposed to a false dichotomy engendered by P values.""")

d(feature_h+feature_d, "content-block")

b(f"""## Get started in seconds

{btn('Install dabest', '/getting_started.ipynb')}""", 'content-block', style={"margin-top": "40px"})






