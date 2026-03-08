using System;
using System.Globalization;
using System.IO;
using UnityEngine;

public class TrajectorySampler : MonoBehaviour
{
    [Header("References")]
    public CarCatcherAgent agent;
    public Camera sourceCamera;

    [Header("Capture")]
    public int width = 64;
    public int height = 64;
    public bool grayscale = true;
    public bool autoStartOnPlay = true;
    public KeyCode toggleCaptureKey = KeyCode.F8;

    [Header("Collection")]
    public int targetTrajectories = 100;
    public int maxFramesPerTrajectory = 0;

    [Header("Output")]
    public string outputRelativePath = "vis_backbone/datasets/manual_capture";
    public bool writePng = true;

    private RenderTexture rt;
    private Texture2D tex;
    private Texture2D preview;
    private string sessionDir;
    private bool isCapturing;
    private bool trajectoryOpen;
    private int completedTrajectories;
    private int trajectoryFrameCount;
    private int observedEpisodeIndex = int.MinValue;
    private StreamWriter csvWriter;

    private void Start()
    {
        if (agent == null)
        {
            agent = FindObjectOfType<CarCatcherAgent>();
        }

        if (sourceCamera == null && agent != null)
        {
            sourceCamera = agent.GetComponentInChildren<Camera>();
        }

        AllocateTextures();
        PrepareSessionDir();

        if (autoStartOnPlay)
        {
            StartCapture();
        }
    }

    private void Update()
    {
        if (Input.GetKeyDown(toggleCaptureKey))
        {
            if (isCapturing) StopCapture();
            else StartCapture();
        }
    }

    private void FixedUpdate()
    {
        if (!Application.isPlaying || !isCapturing || agent == null || sourceCamera == null)
        {
            return;
        }

        if (completedTrajectories >= targetTrajectories)
        {
            StopCapture();
            return;
        }

        int episodeIndex = agent.EpisodeIndex;
        if (episodeIndex < 0)
        {
            return;
        }

        if (observedEpisodeIndex == int.MinValue)
        {
            observedEpisodeIndex = episodeIndex;
            OpenTrajectory();
        }
        else if (episodeIndex != observedEpisodeIndex)
        {
            CloseTrajectory("episode_reset");
            observedEpisodeIndex = episodeIndex;
            if (completedTrajectories >= targetTrajectories)
            {
                StopCapture();
                return;
            }
            OpenTrajectory();
        }

        if (trajectoryOpen)
        {
            int stepCount = agent.StepCount;
            CaptureFrame(stepCount);

            if (maxFramesPerTrajectory > 0 && trajectoryFrameCount >= maxFramesPerTrajectory)
            {
                CloseTrajectory("max_frames");
            }
        }
    }

    private void OnDisable()
    {
        StopCapture();
        CleanupTextures();
    }

    private void StartCapture()
    {
        observedEpisodeIndex = int.MinValue;
        isCapturing = true;
        Debug.Log($"[TrajectorySampler] Capture started. targetTrajectories={targetTrajectories}");
    }

    private void StopCapture()
    {
        isCapturing = false;
        if (trajectoryOpen)
        {
            CloseTrajectory("capture_stopped");
        }
        Debug.Log($"[TrajectorySampler] Capture stopped. completedTrajectories={completedTrajectories}");
    }

    private void PrepareSessionDir()
    {
        string projectRoot = Directory.GetParent(Application.dataPath).FullName;
        string relative = outputRelativePath.Replace('/', Path.DirectorySeparatorChar);
        sessionDir = Path.Combine(projectRoot, relative, DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        Directory.CreateDirectory(sessionDir);
    }

    private void OpenTrajectory()
    {
        string trajName = $"traj_{completedTrajectories:D4}";
        string trajDir = Path.Combine(sessionDir, trajName);
        Directory.CreateDirectory(trajDir);

        string framesDir = Path.Combine(trajDir, "frames");
        Directory.CreateDirectory(framesDir);

        csvWriter = new StreamWriter(Path.Combine(trajDir, "frames.csv"));
        csvWriter.WriteLine(
            "frame_idx,step_count,sim_time,action_fwd,action_lat," +
            "car_px,car_py,car_pz,car_vx,car_vy,car_vz," +
            "ball_px,ball_py,ball_pz,ball_vx,ball_vy,ball_vz");

        trajectoryOpen = true;
        trajectoryFrameCount = 0;
        Debug.Log($"[TrajectorySampler] Open {trajName}");
    }

    private void CloseTrajectory(string reason)
    {
        if (!trajectoryOpen)
        {
            return;
        }

        csvWriter?.Flush();
        csvWriter?.Dispose();
        csvWriter = null;

        string trajName = $"traj_{completedTrajectories:D4}";
        trajectoryOpen = false;
        completedTrajectories += 1;
        trajectoryFrameCount = 0;
        Debug.Log($"[TrajectorySampler] Close {trajName}, reason={reason}, completed={completedTrajectories}/{targetTrajectories}");
    }

    private void CaptureFrame(int stepCount)
    {
        if (rt == null || tex == null || preview == null || rt.width != width || rt.height != height)
        {
            AllocateTextures();
        }

        RenderTexture prev = sourceCamera.targetTexture;
        sourceCamera.targetTexture = rt;
        sourceCamera.Render();
        sourceCamera.targetTexture = prev;

        RenderTexture active = RenderTexture.active;
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
        tex.Apply(false, false);
        RenderTexture.active = active;

        Color32[] pixels = tex.GetPixels32();
        if (grayscale)
        {
            for (int i = 0; i < pixels.Length; i++)
            {
                Color32 p = pixels[i];
                byte g = (byte)(0.299f * p.r + 0.587f * p.g + 0.114f * p.b);
                pixels[i] = new Color32(g, g, g, 255);
            }
        }

        preview.SetPixels32(pixels);
        preview.Apply(false, false);

        string trajDir = Path.Combine(sessionDir, $"traj_{completedTrajectories:D4}");
        if (writePng)
        {
            string framePath = Path.Combine(trajDir, "frames", $"{trajectoryFrameCount:D5}.png");
            File.WriteAllBytes(framePath, preview.EncodeToPNG());
        }

        WriteFrameCsv(stepCount);
        trajectoryFrameCount += 1;
    }

    private void WriteFrameCsv(int stepCount)
    {
        Vector3 carPos = agent.transform.position;
        Vector3 carVel = agent.rb != null ? agent.rb.linearVelocity : Vector3.zero;
        Vector2 action = agent.LastAction;

        Vector3 ballPos = Vector3.zero;
        Vector3 ballVel = Vector3.zero;
        Ball ball = agent.CurrentBall;
        if (ball != null)
        {
            ballPos = ball.transform.position;
            ballVel = ball.rb != null ? ball.rb.linearVelocity : Vector3.zero;
        }

        csvWriter.WriteLine(string.Join(",",
            trajectoryFrameCount.ToString(CultureInfo.InvariantCulture),
            stepCount.ToString(CultureInfo.InvariantCulture),
            Time.time.ToString("F6", CultureInfo.InvariantCulture),
            action.x.ToString("F6", CultureInfo.InvariantCulture),
            action.y.ToString("F6", CultureInfo.InvariantCulture),
            carPos.x.ToString("F6", CultureInfo.InvariantCulture),
            carPos.y.ToString("F6", CultureInfo.InvariantCulture),
            carPos.z.ToString("F6", CultureInfo.InvariantCulture),
            carVel.x.ToString("F6", CultureInfo.InvariantCulture),
            carVel.y.ToString("F6", CultureInfo.InvariantCulture),
            carVel.z.ToString("F6", CultureInfo.InvariantCulture),
            ballPos.x.ToString("F6", CultureInfo.InvariantCulture),
            ballPos.y.ToString("F6", CultureInfo.InvariantCulture),
            ballPos.z.ToString("F6", CultureInfo.InvariantCulture),
            ballVel.x.ToString("F6", CultureInfo.InvariantCulture),
            ballVel.y.ToString("F6", CultureInfo.InvariantCulture),
            ballVel.z.ToString("F6", CultureInfo.InvariantCulture)
        ));
    }

    private void AllocateTextures()
    {
        CleanupTextures();
        rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        rt.filterMode = FilterMode.Point;
        tex = new Texture2D(width, height, TextureFormat.RGB24, false, false);
        preview = new Texture2D(width, height, TextureFormat.RGB24, false, false);
    }

    private void CleanupTextures()
    {
        if (rt != null)
        {
            rt.Release();
            DestroyImmediate(rt);
            rt = null;
        }
        if (tex != null)
        {
            DestroyImmediate(tex);
            tex = null;
        }
        if (preview != null)
        {
            DestroyImmediate(preview);
            preview = null;
        }
    }
}
