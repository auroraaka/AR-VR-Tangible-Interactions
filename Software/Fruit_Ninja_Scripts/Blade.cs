using UnityEngine;

public class Blade : MonoBehaviour
{
    public Vector3 direction { get; private set; }

    private Camera mainCamera;


    private Collider sliceCollider;
    private TrailRenderer sliceTrail;

    public float sliceForce = 5f;
    public float minSliceVelocity = 0.01f;

    private bool slicing;

    public UDPReceive udpReceive;

    private float targetX, targetY;

    private void Awake()
    {
        mainCamera = Camera.main;
        sliceCollider = GetComponent<Collider>();
        sliceTrail = GetComponentInChildren<TrailRenderer>();
    }

    private void OnEnable()
    {
        StopSlice();
    }

    private void OnDisable()
    {
        StopSlice();
    }

    private void Update()
    {
        string data = udpReceive.data;
        data = data.Remove(0, 1);
        data = data.Remove(data.Length - 1, 1);
        Debug.Log("data: " + data);
        string[] points = data.Split(',');

        /// Received values from socket
        targetX = 40 * float.Parse(points[1]) / 100;
        targetY = -20 * float.Parse(points[0]) / 100;

        //Vector3 mousePosition = new Vector2(targetX, targetY);
        //Vector3 newPosition = mainCamera.ScreenToWorldPoint(mousePosition);

        //targetX = newPosition[0];
        //targetY = newPosition[1];

        if (targetX != null)
        {
            IRContinueSlice();
        }

        //if (Input.GetMouseButtonDown(0)) {
        //    StartSlice();
        //} else if (Input.GetMouseButtonUp(0)) {
        //    StopSlice();
        //} else if (slicing) {
        //    ContinueSlice();
        //}
    }

    private void StartSlice()
    {
        Vector3 position = mainCamera.ScreenToWorldPoint(Input.mousePosition);
        position.z = 0f;
        transform.position = position;

        slicing = true;
        sliceCollider.enabled = true;
        sliceTrail.enabled = true;
        sliceTrail.Clear();
    }

    private void StopSlice()
    {
        slicing = false;
        sliceCollider.enabled = false;
        sliceTrail.enabled = false;
    }

    private void ContinueSlice()
    {
        Vector3 newPosition = mainCamera.ScreenToWorldPoint(Input.mousePosition);
        newPosition.z = 0f;

        Debug.Log("mouse: " + newPosition);

        direction = newPosition - transform.position;

        float velocity = direction.magnitude / Time.deltaTime;
        sliceCollider.enabled = velocity > minSliceVelocity;

        transform.position = newPosition;
    }


    // functions for infrared brightmarker detection

    private void IRStartSlice()
    {
        Vector3 position = mainCamera.ScreenToWorldPoint(Input.mousePosition);
        position.z = 0f;
        transform.position = position;

        slicing = true;
        sliceCollider.enabled = true;
        sliceTrail.enabled = true;
        sliceTrail.Clear();
    }

    private void IRStopSlice()
    {
        //slicing = false;
        //sliceCollider.enabled = false;
        //sliceTrail.enabled = false;
    }

    private void IRContinueSlice()
    {
        //Vector3 newPosition = mainCamera.ScreenToWorldPoint(Input.mousePosition);
        //newPosition.z = 0f;

        slicing = true;
        sliceCollider.enabled = true;
        sliceTrail.enabled = true;

        Vector3 newPosition = new Vector3(targetX, targetY, 0);

        Debug.Log("newPosition: " + newPosition);

        direction = newPosition - transform.position;

        float velocity = direction.magnitude / Time.deltaTime;
        sliceCollider.enabled = velocity > minSliceVelocity;

        transform.position = newPosition;
    }

}
