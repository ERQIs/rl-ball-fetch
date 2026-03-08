using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.SideChannels;

public class CarCatcherAgent : Agent
{
    [Header("References")]
    public Rigidbody rb;
    public Transform basketCenter;
    public BallSpawner spawner;

    [Header("Base Control (Continuous)")]
    public float maxForwardSpeed = 8f;
    public float maxLateralSpeed = 8f;
    public bool keepFixedHeading = true;
    public bool useVectorObs = true;

    [Header("Heuristic")]
    [Range(0.1f, 1f)] public float heuristicActionScale = 0.4f;

    [Header("Episode Settings")]
    public float arenaRadius = 6f;
    public int defaultMaxStep = 600;
    public Vector3 episodeStartPosition = new Vector3(0f, 0.2f, -5f);

    [Header("Reward")]
    public float precisionK = 3f;
    // public float progressRewardScale = 10f;
    public float progressRewardScale = 0f;
    public float precisionRewardScale = 20f;
    public float landingProgressRewardScale = 25f;
    public float controlPenaltyScale = 0.02f;
    public float catchReward = 50f;
    public float missReward = -50f;
    public float outOfArenaPenalty = -3f;

    [Header("Debug Reward HUD")]
    public bool showRewardHud = true;
    public int hudFontSize = 20;

    public bool debugNoEndEpisode = false;

    private Ball currentBall;
    private float prevDistance;
    private float prevLandingDistance;
    private Vector2 lastAction;
    private Quaternion fixedRotation;
    private GUIStyle hudStyle;
    private float lastRewardTotal;
    private float lastRewardPos;
    private float lastRewardLanding;
    private float lastRewardPrec;
    private float lastRewardCtrl;
    private float lastRewardTerminal;
    private float lastBallDistance;
    private float lastLandingDistance;
    private int episodeIndex = -1;

    public Vector2 LastAction => lastAction;
    public Ball CurrentBall => currentBall;
    public int EpisodeIndex => episodeIndex;

    public override void Initialize()
    {
        if (rb == null) rb = GetComponent<Rigidbody>();

        fixedRotation = Quaternion.identity;
        rb.centerOfMass = new Vector3(0f, -0.25f, 0f);

        if (keepFixedHeading)
        {
            rb.constraints = RigidbodyConstraints.FreezeRotationX |
                             RigidbodyConstraints.FreezeRotationY |
                             RigidbodyConstraints.FreezeRotationZ;
        }

        if (MaxStep <= 0)
        {
            MaxStep = defaultMaxStep;
        }
    }

    public override void OnEpisodeBegin()
    {
        episodeIndex += 1;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        lastAction = Vector2.zero;

        transform.position = episodeStartPosition;
        transform.rotation = fixedRotation;

        currentBall = spawner.SpawnOrResetBall(this);
        prevDistance = GetBallDistance();
        prevLandingDistance = GetLandingPointDistanceXZ();
        lastRewardTotal = 0f;
        lastRewardPos = 0f;
        lastRewardLanding = 0f;
        lastRewardPrec = 0f;
        lastRewardCtrl = 0f;
        lastRewardTerminal = 0f;
        lastBallDistance = prevDistance;
        lastLandingDistance = prevLandingDistance;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!useVectorObs)
        {
            return;
        }

        if (currentBall == null)
        {
            sensor.AddObservation(Vector3.zero); // delta_pos
            sensor.AddObservation(Vector3.zero); // ball_vel
            sensor.AddObservation(Vector3.zero); // base_vel
            sensor.AddObservation(Vector3.zero); // dir_to_ball
            return;
        }

        Vector3 deltaPos = currentBall.transform.position - basketCenter.position;
        Vector3 ballVel = currentBall.rb.linearVelocity;
        Vector3 baseVel = rb.linearVelocity;
        Vector3 dirToBall = deltaPos.sqrMagnitude > 1e-6f ? deltaPos.normalized : Vector3.zero;

        sensor.AddObservation(deltaPos);  // 3
        sensor.AddObservation(ballVel);   // 3
        sensor.AddObservation(baseVel);   // 3
        sensor.AddObservation(dirToBall); // 3
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // a = [v_forward, v_lateral], each in [-1, 1]
        float forwardCmd = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float lateralCmd = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);
        lastAction = new Vector2(forwardCmd, lateralCmd);

        Vector3 targetPlanarVel =
            transform.forward * (forwardCmd * maxForwardSpeed) +
            transform.right * (lateralCmd * maxLateralSpeed);

        rb.linearVelocity = new Vector3(targetPlanarVel.x, rb.linearVelocity.y, targetPlanarVel.z);

        if (keepFixedHeading)
        {
            transform.rotation = fixedRotation;
            rb.angularVelocity = Vector3.zero;
        }

        if (currentBall != null)
        {
            float d = GetBallDistance();
            float rPos = prevDistance - d;
            float landingDistance = GetLandingPointDistanceXZ();
            float rLandingPos = prevLandingDistance - landingDistance;
            float rPrec = Mathf.Exp(-precisionK * d);
            float rCtrl = -controlPenaltyScale * lastAction.sqrMagnitude;
            float rewardPosTerm = progressRewardScale * rPos;
            float rewardLandingTerm = landingProgressRewardScale * rLandingPos;
            float rewardPrecTerm = precisionRewardScale * rPrec;
            float rewardTotal = rewardPosTerm + rewardLandingTerm + rewardPrecTerm + rCtrl;

            AddReward(rewardTotal);
            prevDistance = d;
            prevLandingDistance = landingDistance;
            lastRewardPos = rewardPosTerm;
            lastRewardLanding = rewardLandingTerm;
            lastRewardPrec = rewardPrecTerm;
            lastRewardCtrl = rCtrl;
            lastRewardTerminal = 0f;
            lastRewardTotal = rewardTotal;
            lastBallDistance = d;
            lastLandingDistance = landingDistance;
        }
        else
        {
            // Small penalty if no target exists in this step.
            AddReward(-0.001f);
            lastRewardPos = 0f;
            lastRewardLanding = 0f;
            lastRewardPrec = 0f;
            lastRewardCtrl = 0f;
            lastRewardTerminal = 0f;
            lastRewardTotal = -0.001f;
        }

        if (IsOutOfArena())
        {
            AddReward(outOfArenaPenalty);
            lastRewardTerminal = outOfArenaPenalty;
            lastRewardTotal += outOfArenaPenalty;
            if (!debugNoEndEpisode) EndEpisode();
        }
    }

    public void OnBallCaught()
    {
        Academy.Instance.StatsRecorder.Add("CarCatch/SuccessRate", 1f, StatAggregationMethod.Average);
        AddReward(catchReward);
        lastRewardTerminal = catchReward;
        lastRewardTotal += catchReward;
        if (!debugNoEndEpisode) EndEpisode();
    }

    public void OnBallMissed()
    {
        Academy.Instance.StatsRecorder.Add("CarCatch/SuccessRate", 0f, StatAggregationMethod.Average);
        AddReward(missReward);
        lastRewardTerminal = missReward;
        lastRewardTotal += missReward;
        if (!debugNoEndEpisode) EndEpisode();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        ca[0] = 0f; // forward/backward
        ca[1] = 0f; // left/right

        if (Input.GetKey(KeyCode.W)) ca[0] += 1f;
        if (Input.GetKey(KeyCode.S)) ca[0] -= 1f;
        if (Input.GetKey(KeyCode.D)) ca[1] += 1f;
        if (Input.GetKey(KeyCode.A)) ca[1] -= 1f;

        float scale = Mathf.Clamp(heuristicActionScale, 0.1f, 1f);
        ca[0] = Mathf.Clamp(ca[0], -1f, 1f) * scale;
        ca[1] = Mathf.Clamp(ca[1], -1f, 1f) * scale;
    }

    private float GetBallDistance()
    {
        if (currentBall == null) return 0f;
        return Vector3.Distance(currentBall.transform.position, basketCenter.position);
    }

    private float GetLandingPointDistanceXZ()
    {
        if (currentBall == null) return 0f;

        Vector3 a = basketCenter.position;
        Vector3 b = currentBall.predictedLandingPoint;
        a.y = 0f;
        b.y = 0f;
        return Vector3.Distance(a, b);
    }

    private bool IsOutOfArena()
    {
        Vector3 p = transform.position;
        Vector2 planar = new Vector2(p.x, p.z);
        return planar.magnitude > arenaRadius;
    }

    private void OnGUI()
    {
        if (!showRewardHud || !Application.isPlaying)
        {
            return;
        }

        if (hudStyle == null)
        {
            hudStyle = new GUIStyle(GUI.skin.box);
            hudStyle.fontSize = hudFontSize;
            hudStyle.alignment = TextAnchor.UpperLeft;
            hudStyle.normal.textColor = Color.white;
        }

        string text =
            $"Step: {StepCount}\n" +
            $"Cumulative: {GetCumulativeReward():F3}\n" +
            $"LastTotal: {lastRewardTotal:F3}\n" +
            $"r_pos: {lastRewardPos:F3}\n" +
            $"r_land: {lastRewardLanding:F3}\n" +
            $"r_prec: {lastRewardPrec:F3}\n" +
            $"r_ctrl: {lastRewardCtrl:F3}\n" +
            $"r_terminal: {lastRewardTerminal:F3}\n" +
            $"ball_d: {lastBallDistance:F3}\n" +
            $"land_d: {lastLandingDistance:F3}";

        GUI.Box(new Rect(10f, 10f, 320f, 280f), text, hudStyle);
    }
}
