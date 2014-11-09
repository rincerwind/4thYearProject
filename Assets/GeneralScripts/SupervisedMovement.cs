using UnityEngine;
using System.Collections;
using LA = MathNet.Numerics.LinearAlgebra;

public class SupervisedMovement : MonoBehaviour {

	public float maxSpeed;
	public float moveSpeed;
	public float acceleration;
	public bool recordMovement;
	public bool debugMovement;

	private NeuralNetwork[] nets;
	private ArrayList targetValues;
	private ArrayList initialInputs;
	private GameObject target;
	private NextGoal g;

	// Use this for initialization
	void Start () {
		nets = gameObject.GetComponents<NeuralNetwork> ();
		target = GameObject.FindGameObjectWithTag("Goal");
		targetValues = new ArrayList ();
		initialInputs = new ArrayList ();
		g = target.GetComponent<NextGoal>();
	}

	// Update is called once per frame
	void FixedUpdate () {
		LA.Matrix<float> inputs;
		LA.Matrix<float> outputs;
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		Vector3 direction = new Vector3 (horizontalMovement, 0, verticalMovement);

		// Recording Phase
		if (recordMovement && !nets[0].TrainingPhase) {
			float deltaX = target.transform.position.x - transform.position.x;
			float deltaZ = target.transform.position.z - transform.position.z;

			targetValues.Add (horizontalMovement);
			targetValues.Add (verticalMovement);
			initialInputs.Add (deltaX);
			initialInputs.Add (deltaZ);
		}

		// Learning Phase
		if ( !recordMovement && nets[0].TrainingPhase ) {
			nets[0].LearningPhase(initialInputs, targetValues, nets[0].allowedError);
			nets[0].TrainingPhase = false;
		}

		inputs = LA.Matrix<float>.Build.Dense (1, nets[0].numInputs, new float[]{
			target.transform.position.x - transform.position.x,
			target.transform.position.z - transform.position.z});

		// Neural Net in action
		if ( !debugMovement && !recordMovement && !nets[0].TrainingPhase ){
			outputs = nets[0].ComputeOutputs(inputs);
			direction.x = outputs[0,0];
			direction.z = outputs[0,1];
		}

		if( rigidbody.velocity.magnitude < maxSpeed )
			rigidbody.AddForce (transform.rotation * direction * moveSpeed);
	}// end of FixedUpdate

	void OnTriggerEnter(Collider c){
		switch (WorldManager.currentLevel){
			case 0:
				if( c.transform.tag == "Goal" ){
					if( g == null || ( g != null && g.isLastGoal() ) )
						WorldManager.CompleteLevel();
					else{
						g.goToNextGoal();
						target.transform.position = (g.getCurrentGoal()).position;
					}
				}
				break;
			default:
				WorldManager.CompleteLevel();
				break;
		}
	}
}// end of class
