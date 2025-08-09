import os

folder = r"C:\Users\Aafrin\Documents\PROJECTS\Useless Project\backend\sample-leaves"
print("Looking in folder:", folder)
print("os.path.exists:", os.path.exists(folder))
print("os.path.isdir:", os.path.isdir(folder))

if not os.path.isdir(folder):
    print("Folder does NOT exist or is not a directory!")
else:
    print("Folder exists and is a directory.")

for filename in os.listdir(folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(folder, filename)
        print(f"Found image: {filename}")
