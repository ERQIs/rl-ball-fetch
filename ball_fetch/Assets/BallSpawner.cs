using UnityEngine;

public class BallSpawner : MonoBehaviour
{
    public Ball ballPrefab;

    [Header("Spawn")]
    public Vector3 fixedSpawnPosition = new Vector3(0f, 1.5f, 20f);

    [Header("Landing Sector (relative to car)")]
    public float landingMinRadius = 3f;
    public float landingMaxRadius = 6f;
    public float landingSectorAngleDeg = 120f; // centered on car forward
    public float landingHeightOffset = 0f;

    [Header("Flight Time Range")]
    public float Tmin = 1.0f;
    public float Tmax = 1.8f;

    [Header("Debug")]
    public bool showLandingMarker = true;
    public float markerScale = 0.35f;
    public Color markerColor = Color.red;

    private Ball ballInstance;
    private GameObject landingMarker;

    public Ball SpawnOrResetBall(CarCatcherAgent agent)
    {
        if (ballInstance == null)
        {
            ballInstance = Instantiate(ballPrefab);
        }

        ballInstance.agent = agent;

        Vector3 spawnPos = fixedSpawnPosition;
        ballInstance.transform.position = spawnPos;
        ballInstance.rb.linearVelocity = Vector3.zero;
        ballInstance.rb.angularVelocity = Vector3.zero;

        Vector3 target = SampleLandingPoint(agent);

        float T = Random.Range(Tmin, Tmax);
        float gravityScale = ballInstance != null ? ballInstance.gravityScale : 1f;
        Vector3 g = Physics.gravity * gravityScale;
        Vector3 v0 = (target - spawnPos - 0.5f * g * T * T) / T;

        ballInstance.predictedLandingPoint = target;
        ballInstance.predictedT = T;
        ballInstance.ResetLandingCheck();
        ballInstance.rb.linearVelocity = v0;
        UpdateLandingMarker(target);

        return ballInstance;
    }

    private Vector3 SampleLandingPoint(CarCatcherAgent agent)
    {
        float halfAngle = landingSectorAngleDeg * 0.5f;
        float angle = Random.Range(-halfAngle, halfAngle);
        float radius = Random.Range(landingMinRadius, landingMaxRadius);

        Quaternion yawRot = Quaternion.Euler(0f, angle, 0f);
        Vector3 dir = yawRot * agent.transform.forward;
        dir.y = 0f;
        dir.Normalize();

        Vector3 center = agent.transform.position;
        center.y = agent.basketCenter.position.y + landingHeightOffset;

        return center + dir * radius;
    }

    private void UpdateLandingMarker(Vector3 target)
    {
        if (!showLandingMarker)
        {
            if (landingMarker != null)
            {
                landingMarker.SetActive(false);
            }
            return;
        }

        if (landingMarker == null)
        {
            landingMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            landingMarker.name = "PredictedLandingMarker";

            Collider col = landingMarker.GetComponent<Collider>();
            if (col != null) col.enabled = false;

            Rigidbody rb = landingMarker.AddComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity = false;

            Renderer renderer = landingMarker.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = markerColor;
            }
        }

        landingMarker.SetActive(true);
        landingMarker.transform.position = target;
        landingMarker.transform.localScale = Vector3.one * markerScale;

        Renderer markerRenderer = landingMarker.GetComponent<Renderer>();
        if (markerRenderer != null)
        {
            markerRenderer.material.color = markerColor;
        }
    }
}
