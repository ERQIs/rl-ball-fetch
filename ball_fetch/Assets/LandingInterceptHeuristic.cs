using UnityEngine;

[DisallowMultipleComponent]
[RequireComponent(typeof(CarCatcherAgent))]
public class LandingInterceptHeuristic : MonoBehaviour
{
    [Header("References")]
    public CarCatcherAgent agent;

    [Header("Timing")]
    public float minRemainingTime = 0.05f;
    [Range(0f, 1f)] public float motionStartFraction = 0.5f;

    [Header("Behavior")]
    public float stopDistance = 0.05f;
    public bool zeroActionWhenNoBall = true;
    public bool drawDebugLine = true;

    private void Awake()
    {
        if (agent == null)
        {
            agent = GetComponent<CarCatcherAgent>();
        }
    }

    public bool TryComputeAction(out Vector2 action)
    {
        action = Vector2.zero;

        if (agent == null || agent.basketCenter == null)
        {
            return false;
        }

        Ball ball = agent.CurrentBall;
        if (ball == null)
        {
            return zeroActionWhenNoBall;
        }

        float motionGateTime = Mathf.Max(0f, ball.predictedT * motionStartFraction);
        if (ball.ElapsedFlightTime < motionGateTime)
        {
            return true;
        }

        Vector3 targetOffset = ball.predictedLandingPoint - agent.basketCenter.position;
        targetOffset.y = 0f;
        if (targetOffset.sqrMagnitude <= stopDistance * stopDistance)
        {
            return true;
        }

        float remainingTime = Mathf.Max(minRemainingTime, ball.RemainingFlightTime);
        Vector3 requiredWorldVel = targetOffset / remainingTime;
        Vector3 requiredLocalVel = agent.transform.InverseTransformDirection(requiredWorldVel);

        float forwardCmd = 0f;
        if (agent.maxForwardSpeed > 1e-5f)
        {
            forwardCmd = requiredLocalVel.z / agent.maxForwardSpeed;
        }

        float lateralCmd = 0f;
        if (agent.maxLateralSpeed > 1e-5f)
        {
            lateralCmd = requiredLocalVel.x / agent.maxLateralSpeed;
        }

        action = new Vector2(
            Mathf.Clamp(forwardCmd, -1f, 1f),
            Mathf.Clamp(lateralCmd, -1f, 1f)
        );

        if (drawDebugLine)
        {
            Debug.DrawLine(agent.basketCenter.position, ball.predictedLandingPoint, Color.cyan, Time.fixedDeltaTime);
        }

        return true;
    }
}
