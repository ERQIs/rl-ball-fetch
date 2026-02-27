using UnityEngine;

public class Ball : MonoBehaviour
{
    public Rigidbody rb;
    [HideInInspector] public CarCatcherAgent agent;

    [HideInInspector] public Vector3 predictedLandingPoint;
    [HideInInspector] public float predictedT;

    private bool landed = false;

    public float gravityScale = 0.35f; // 1=正常，0.35=慢很多

    private void Awake()
    {
        if (rb == null) rb = GetComponent<Rigidbody>();
    }

    private void FixedUpdate()
    {
        // 把重力改成 gravityScale 倍（只影响球）
        rb.AddForce(Physics.gravity * (gravityScale - 1f), ForceMode.Acceleration);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (landed) return;

        if (collision.collider.CompareTag("Ground"))
        {
            landed = true;

            Vector3 actual = transform.position;
            float err = Vector3.Distance(
                new Vector3(actual.x, 0f, actual.z),
                new Vector3(predictedLandingPoint.x, 0f, predictedLandingPoint.z)
            );

            Debug.Log($"[LandingCheck] T={predictedT:F2}s  predicted={predictedLandingPoint}  actual={actual}  horiz_err={err:F3}m");

            agent?.OnBallMissed(); // 你想调试不重置就先注释掉
        }
    }


    public void ResetLandingCheck()
    {
        landed = false;
    }
}