# Handling Occlusion: Disappearance Context and Ideas

Your understanding is right. The skeleton file records only "no person in this frame". It loses how the person disappeared: going down fast, in the middle of the room, next to the bed, and never coming back.

I measured how strong that context is by looking at every time someone disappeared for at least a second after being clearly visible:

| | Disappearances | Median distance to frame edge | Disappeared mid-frame | Downward hip speed just before |
|---|---|---|---|---|
| OCCU, during a fall | 60 | 0.35 | 98% | 0.32 frame heights/s |
| OCCU, not a fall | 10 | 0.11 | 50% | 0.09 |
| Le2i, during a fall | 11 | 0.31 | 100% | 0.14 |
| Le2i, not a fall | 33 | 0.12 | 58% | 0.01 |

The context separates cleanly. Someone who falls behind furniture disappears mid-frame while dropping fast. Someone who leaves disappears near the edge while moving sideways. So this is solvable, and OCCU becomes useful training data instead of something to throw away.

Ideas, roughly from cheapest to most ambitious:

## Using the skeleton data we already have

### 1. Treat "disappeared" as information, not a gap

Give the model a visible/not-visible channel for each frame, plus "last seen" features carried forward through the gap:

- time since last seen
- last hip height and downward speed
- distance to the frame edge when lost

The model can then learn that "dropped fast mid-frame, then gone" means fell, while "walked to the edge, then gone" means left. This also fixes my earlier worry that "no skeleton means fallen" would trigger alarms whenever someone walks out of view, because walking out looks different. OCCU's train clips can then be used, rather than excluded by a QC rule.

### 2. A tracker that reasons about disappearances, as a safety net

A small rule-based tracker that runs alongside the model, with three states:

- **visible**
- **left the frame** (lost near an edge or doorway)
- **hidden after going down** (lost mid-frame after a fast drop)

If someone is hidden after going down and doesn't reappear within N seconds, it raises a CHECK alert. That fits the CHECK band already in the plan. It needs no training and can be tested on OCCU's test clips straight away.

### 3. Simulate occlusion during training

Le2i and OF-Syn have hundreds of falls where the person stays visible. During training, randomly blank out the skeleton from some point after the fall starts, as if furniture appeared, and add random boxes for partial occlusion. The model then learns "fall, then vanish" from thousands of examples instead of 60, and becomes more robust to partly hidden people in general.

### 4. Predict where the person should be

A motion predictor (a Kalman filter) continues the person's path through the gap. If the predicted position is still inside the frame and heading toward the floor, they didn't leave; they're hidden. This is the "tracking paths" idea you suggested.

## Better perception

### 5. Learn where the furniture is

For each camera, learn the "blind spots": places where people regularly disappear. Alternatively, segment the bed and sofa once at setup with a segmentation model (for example SAM). Disappearing next to a known blind spot at floor height is strong evidence of a hidden fall.

### 6. Use models that handle partly visible bodies

MediaPipe tends to return nothing unless it sees most of a body. A person detector followed by a pose model (for example YOLO-pose or RTMPose) still returns keypoints for a head and shoulders. This doesn't help when the person is fully hidden, as in OCCU, but real homes have more partial than full occlusion.

### 7. Watch for motion behind the furniture

Measure how much the pixels behind the bed change after a fall, without storing any images. Movement there means someone is down but moving; no movement after a fall is the most urgent case. This keeps the "skeleton only, no video stored" privacy promise.

## Beyond a single camera

### 8. A second camera angle

Each OCCU actor is filmed from both view1 and view2, so a person hidden in one angle may be visible from the other. Combine the two views and alert if either sees a fall. The live demo could use two webcams. It's very effective, and OCCU already provides the data to test it.

### 9. Other sensors

mmWave radar or thermal sensors can detect a person behind furniture, depth cameras see over it, and a microphone can pick up the sound of the impact. These are out of scope for the hackathon; mention them in the README's future-work section.

## What I'd recommend for the hackathon

Do **1 + 3** (the model learns from disappearance context and simulated occlusion) and **2** (the tracker as a safety net producing CHECK alerts), and report OCCU test results with and without them. Add **8** if there's time.

## What this changes

- **Contract 3 (Dev B's window files):** gains "visible" and "last seen" features. Contracts haven't been agreed yet (S3), so now is the cheapest time to change them.
- **QC:** OCCU gets an occlusion-aware rule instead of being excluded.
- **Who does what:** Dev B owns the windowing, features and model (1 and 3). I'd build the tracker (2), the QC change and the OCCU evaluation.
