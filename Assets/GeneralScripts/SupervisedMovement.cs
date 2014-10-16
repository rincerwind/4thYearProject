using UnityEngine;
using System.Collections;
using LA = MathNet.Numerics.LinearAlgebra;

public class SupervisedMovement : MonoBehaviour {

	public float maxSpeed;
	public float moveSpeed;
	public float acceleration;
	public bool recordMovement;

	private Vector3 direction;
	private bool debug;
	private NeuralNetwork n;

	// Use this for initialization
	void Start () {
		debug = false;
		recordMovement = false;
		n = gameObject.GetComponent<NeuralNetwork> ();
	}
	
	// Update is called once per frame
	void FixedUpdate () {
		direction = new Vector3 (Input.GetAxis("Horizontal"), 0, Input.GetAxis("Vertical"));

		
		if( rigidbody.velocity.magnitude < maxSpeed )
			rigidbody.AddForce (direction * moveSpeed);
		
		//if (transform.position.y < -1)
		//	player_die ();
		
		if( debug == true )
			Debug.Log (direction);

		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, 3, new float[] {1.0f, 2.0f, 3.0f});
		n.ComputeOutputs (inputs);

		LA.Matrix<float> outputs = LA.Matrix<float>.Build.Dense (1, 2, new float[]{-0.875f, 0.7275f});
		n.UpdateWeights (outputs, 0.9f, 1.0f);
	}
}
