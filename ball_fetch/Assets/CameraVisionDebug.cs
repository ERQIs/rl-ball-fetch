using System.IO;
using UnityEngine;

[ExecuteAlways]
public class CameraVisionDebug : MonoBehaviour
{
    [Header("Source")]
    public Camera sourceCamera;

    [Header("Output")]
    public int width = 64;
    public int height = 64;
    public bool grayscale = true;
    public int captureEveryNFrames = 10;
    public bool showOnScreen = true;
    public bool savePng = false;

    [Header("Overlay")]
    public int overlaySize = 160;
    public int overlayPadding = 10;

    RenderTexture _rt;
    Texture2D _tex;
    Texture2D _preview;
    int _frame;

    void OnEnable()
    {
        if (sourceCamera == null)
        {
            sourceCamera = GetComponent<Camera>();
        }
        Allocate();
    }

    void OnDisable()
    {
        Cleanup();
    }

    void Allocate()
    {
        Cleanup();
        _rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        _rt.filterMode = FilterMode.Point;
        _tex = new Texture2D(width, height, TextureFormat.RGB24, false, false);
        _preview = new Texture2D(width, height, TextureFormat.RGB24, false, false);
    }

    void Cleanup()
    {
        if (_rt != null)
        {
            _rt.Release();
            DestroyImmediate(_rt);
            _rt = null;
        }
        if (_tex != null)
        {
            DestroyImmediate(_tex);
            _tex = null;
        }
        if (_preview != null)
        {
            DestroyImmediate(_preview);
            _preview = null;
        }
    }

    void LateUpdate()
    {
        if (sourceCamera == null)
        {
            return;
        }
        if (width <= 0 || height <= 0)
        {
            return;
        }
        if (_rt == null || _tex == null || _preview == null || _rt.width != width || _rt.height != height)
        {
            Allocate();
        }

        _frame++;
        if (captureEveryNFrames > 1 && (_frame % captureEveryNFrames) != 0)
        {
            return;
        }

        var prev = sourceCamera.targetTexture;
        sourceCamera.targetTexture = _rt;
        sourceCamera.Render();
        sourceCamera.targetTexture = prev;

        var active = RenderTexture.active;
        RenderTexture.active = _rt;
        _tex.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
        _tex.Apply(false, false);
        RenderTexture.active = active;

        if (grayscale)
        {
            var pixels = _tex.GetPixels32();
            for (int i = 0; i < pixels.Length; i++)
            {
                var p = pixels[i];
                byte g = (byte)(0.299f * p.r + 0.587f * p.g + 0.114f * p.b);
                pixels[i] = new Color32(g, g, g, 255);
            }
            _preview.SetPixels32(pixels);
        }
        else
        {
            _preview.SetPixels32(_tex.GetPixels32());
        }
        _preview.Apply(false, false);

        if (savePng)
        {
            var bytes = _preview.EncodeToPNG();
            var dir = Path.Combine(Application.dataPath, "DebugCaptures");
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, $"carcam_{System.DateTime.Now:yyyyMMdd_HHmmss_fff}.png");
            File.WriteAllBytes(path, bytes);
        }
    }

    void OnGUI()
    {
        if (!showOnScreen || _preview == null)
        {
            return;
        }
        int x = Screen.width - overlayPadding - overlaySize;
        int y = overlayPadding;
        GUI.DrawTexture(new Rect(x, y, overlaySize, overlaySize), _preview, ScaleMode.ScaleToFit, false);
    }
}
