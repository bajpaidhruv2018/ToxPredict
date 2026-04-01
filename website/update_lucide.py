import os

html_path = "c:/College/hackathons/CodeCure/ToxPredict/website/index.html"
js_path = "c:/College/hackathons/CodeCure/ToxPredict/website/script.js"

for path in [html_path, js_path]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace("alert-triangle", "triangle-alert")
    content = content.replace("alert-octagon", "octagon-alert")
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Updated lucide deprecated names.")
