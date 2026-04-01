import os
import re

html_path = "c:/College/hackathons/CodeCure/ToxPredict/website/index.html"
js_path = "c:/College/hackathons/CodeCure/ToxPredict/website/script.js"

with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Emojis dictionary
replacements = {
    # Headings and logos
    "🧬 ToxPredict": "<i data-lucide=\"dna\" class=\"icon-inline\"></i> ToxPredict",
    "🏆 CodeCure": "<i data-lucide=\"award\" class=\"icon-inline\"></i> CodeCure",
    "🔬 Try ToxPredict": "<i data-lucide=\"microscope\" class=\"icon-inline\"></i> Try ToxPredict",
    "🔬 Predict": "<i data-lucide=\"microscope\" class=\"icon-inline\"></i> Predict",
    
    # Problem cards
    "<div class=\"problem-icon\">⏰</div>": "<div class=\"problem-icon\"><i data-lucide=\"clock\" style=\"width:32px; height:32px;\"></i></div>",
    "<div class=\"problem-icon\">💀</div>": "<div class=\"problem-icon\"><i data-lucide=\"skull\" style=\"width:32px; height:32px;\"></i></div>",
    "<div class=\"problem-icon\">💰</div>": "<div class=\"problem-icon\"><i data-lucide=\"circle-dollar-sign\" style=\"width:32px; height:32px;\"></i></div>",
    
    # How it works
    "<div class=\"step-icon\">🧪</div>": "<div class=\"step-icon\"><i data-lucide=\"flask-conical\"></i></div>",
    "<div class=\"step-icon\">🧬</div>": "<div class=\"step-icon\"><i data-lucide=\"dna\"></i></div>",
    "<div class=\"step-icon\">🤖</div>": "<div class=\"step-icon\"><i data-lucide=\"cpu\"></i></div>",
    "<div class=\"step-icon\">⚠️</div>": "<div class=\"step-icon\"><i data-lucide=\"alert-triangle\"></i></div>",
    "<div class=\"step-icon\">📋</div>": "<div class=\"step-icon\"><i data-lucide=\"file-text\"></i></div>",

    # Tabs
    "🔤 Search by Drug Name": "<i data-lucide=\"type\" class=\"icon-sm\"></i> Search by Drug Name",
    "🧪 Enter SMILES": "<i data-lucide=\"test-tube\" class=\"icon-sm\"></i> Enter SMILES",
    "⚖️ Compare Drugs": "<i data-lucide=\"scale\" class=\"icon-sm\"></i> Compare Drugs",
    "⚖️ Compare": "<i data-lucide=\"scale\" class=\"icon-sm\"></i> Compare",

    # Quick pills
    "💊 Aspirin": "<i data-lucide=\"pill\" class=\"icon-sm\"></i> Aspirin",
    "🧪 Estradiol": "<i data-lucide=\"flask-conical\" class=\"icon-sm\"></i> Estradiol",
    "☕ Caffeine": "<i data-lucide=\"coffee\" class=\"icon-sm\"></i> Caffeine",
    "💪 Testosterone": "<i data-lucide=\"activity\" class=\"icon-sm\"></i> Testosterone",
    "⚠️ Thalidomide": "<i data-lucide=\"alert-triangle\" class=\"icon-sm\"></i> Thalidomide",
    "🔴 Doxorubicin": "<i data-lucide=\"alert-octagon\" class=\"icon-sm\"></i> Doxorubicin",

    # Features
    "<div class=\"feature-icon cyan\">🔬</div>": "<div class=\"feature-icon cyan\"><i data-lucide=\"box\" style=\"width:32px; height:32px;\"></i></div>",
    "<div class=\"feature-icon purple\">🎯</div>": "<div class=\"feature-icon purple\"><i data-lucide=\"target\" style=\"width:32px; height:32px;\"></i></div>",
    "<div class=\"feature-icon red\">⚠️</div>": "<div class=\"feature-icon red\"><i data-lucide=\"alert-triangle\" style=\"width:32px; height:32px;\"></i></div>",
    "<div class=\"feature-icon green\">💊</div>": "<div class=\"feature-icon green\"><i data-lucide=\"activity\" style=\"width:32px; height:32px;\"></i></div>",
    "<div class=\"feature-icon amber\">⚖️</div>": "<div class=\"feature-icon amber\"><i data-lucide=\"scale\" style=\"width:32px; height:32px;\"></i></div>",
    "<div class=\"feature-icon blue\">📋</div>": "<div class=\"feature-icon blue\"><i data-lucide=\"layers\" style=\"width:32px; height:32px;\"></i></div>",

    # Rest
    "⚠️ HIGH RISK": "<i data-lucide=\"alert-triangle\" class=\"icon-inline\"></i> HIGH RISK",
    "⚠": "<i data-lucide=\"alert-triangle\" style=\"width: 14px; height: 14px;\"></i>",
    "✅": "<i data-lucide=\"check-circle\" style=\"width: 14px; height: 14px;\"></i>",
    "❌": "<i data-lucide=\"x-circle\" style=\"width: 14px; height: 14px;\"></i>",
    "🔍": "<i data-lucide=\"search\"></i>",
    "📥": "<i data-lucide=\"download\"></i>",
    "✓": "<i data-lucide=\"check\" style=\"width:12px; height:12px; display:inline-block;\"></i>",
    "✗": "<i data-lucide=\"x\" style=\"width:12px; height:12px; display:inline-block;\"></i>"
}

for old, new in replacements.items():
    html = html.replace(old, new)

# Add lucide CDN missing
if "lucide@latest" not in html:
    html = html.replace("</head>", "  <script src=\"https://unpkg.com/lucide@latest\"></script>\n</head>")

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)

print("Updated HTML.")

with open(js_path, 'r', encoding='utf-8') as f:
    js = f.read()

# Update JS strings that use emojis
js = js.replace("⚠️ HIGH RISK", "<i data-lucide=\\'alert-triangle\\' class=\\'icon-inline\\'></i> HIGH RISK")
js = js.replace("⚡ MODERATE RISK", "<i data-lucide=\\'alert-circle\\' class=\\'icon-inline\\'></i> MODERATE RISK")
js = js.replace("✅ LOW RISK", "<i data-lucide=\\'check-circle\\' class=\\'icon-inline\\'></i> LOW RISK")
js = js.replace("'✅'", "'<i data-lucide=\"check-circle\" class=\"icon-inline\"></i>'")
js = js.replace("'⚠️'", "'<i data-lucide=\"alert-triangle\" class=\"icon-inline\"></i>'")

# Special fix for string interpolations using emojis
# e.g.: <span>${isGood ? '✅' : '⚠️'} ${label}</span>
js = js.replace("${isGood ? '✅' : '⚠️'}", "${isGood ? '<i data-lucide=\"check-circle\" class=\"icon-sm\"></i>' : '<i data-lucide=\"alert-triangle\" class=\"icon-sm\"></i>'}")

js = js.replace("<span class=\"tox-alert-icon\">⚠️</span>", "<i data-lucide=\"alert-triangle\" class=\"tox-alert-icon\"></i>")
js = js.replace("✅ No known toxicophores detected", "<i data-lucide=\"check-circle\" class=\"icon-inline\"></i> No known toxicophores detected")
js = js.replace("❌", "<i data-lucide=\"x-circle\" class=\"icon-inline\"></i>")

# add lucide initialization to init()
if "lucide.createIcons();" not in js:
    js = js.replace("initHamburger();", "initHamburger();\n  lucide.createIcons();")

with open(js_path, 'w', encoding='utf-8') as f:
    f.write(js)

print("Updated JS.")
