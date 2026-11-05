with open("app.py", "r") as f:
    content = f.read()

target = """            # H. Bottom Assistant Response Overlay
            if active_response:
                render_response_overlay(frame, active_response)"""

replacement = """            # H. Top-Right Obstacle Warning & Bottom Assistant Response Overlays
            if obstacle_warnings:
                render_obstacle_overlay(frame, obstacle_warnings[0])

            if active_response:
                render_response_overlay(frame, active_response)"""

if target in content:
    content = content.replace(target, replacement)
    with open("app.py", "w") as f:
        f.write(content)
    print("Fixed overlay in app.py!")
else:
    print("Target string not found!")
