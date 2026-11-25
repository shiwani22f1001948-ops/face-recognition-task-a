"""Build practice data from the public LFW dataset (http://vis-www.cs.umass.edu/lfw/).
Download + extract lfw.tgz once (this is setup, not runtime), then:

    python tools/make_demo_data.py path/to/lfw

Creates  demo_data/people/<Name>.jpg   (50 people, one photo each)
         demo_data/group_1.jpg ...     (10 faces: 7 people from the folder using a DIFFERENT photo
                                        of them + 3 people who are NOT in the folder)
LFW photos are 250x250 head shots, so the "group photo" is a collage - fine for a smoke test, but
also try real group photos (phone photos of friends/family who also appear in your folder)."""
import os, random, sys
from PIL import Image

root = sys.argv[1]
random.seed(7)
people = {d: sorted(os.listdir(os.path.join(root, d))) for d in sorted(os.listdir(root))
          if os.path.isdir(os.path.join(root, d))}
multi = [d for d, f in people.items() if len(f) >= 2]
random.shuffle(multi)
known, strangers = multi[:50], multi[50:80]
out = "demo_data"; os.makedirs(os.path.join(out, "people"), exist_ok=True)
for d in known:
    Image.open(os.path.join(root, d, people[d][0])).convert("RGB").save(
        os.path.join(out, "people", d.replace("_", " ") + ".jpg"))
for g in range(1, 4):
    pick = random.sample(known, 7)
    tiles = [Image.open(os.path.join(root, d, people[d][1])) for d in pick]
    tiles += [Image.open(os.path.join(root, d, people[d][0])) for d in random.sample(strangers, 3)]
    random.shuffle(tiles)
    sheet = Image.new("RGB", (250 * 5, 250 * 2), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t.convert("RGB").resize((250, 250)), ((i % 5) * 250, (i // 5) * 250))
    sheet.save(os.path.join(out, f"group_{g}.jpg"))
    print(f"group_{g}.jpg  known: {', '.join(sorted(pick))}")
print("Done ->", out)
