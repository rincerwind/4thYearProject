using UnityEngine;
using System.Collections;
using LA = MathNet.Numerics.LinearAlgebra;

public class NetworkController : MonoBehaviour {

	public bool recordMovement;
	public bool debugMovement;
	public bool TrainingPhase;

	private NeuralNetwork[] nets;
	private GameObject target;
	private SupervisedMotion sm;
	private RadialGridSensor sensor;

	// Use this for initialization
	public void Start () {
		nets = gameObject.GetComponents<NeuralNetwork> ();
		sm = gameObject.GetComponent<SupervisedMotion>();
		target = GameObject.FindGameObjectWithTag("Goal");
		sensor = gameObject.transform.GetChild(0).gameObject.GetComponent<RadialGridSensor> ();
	}

	private float vectorDiff(Vector3 targetDirection, Vector3 lookDirection){
			
			// the vector perpendicular to referenceForward (90 degrees clockwise)
			// (used to determine if angle is positive or negative)
		Vector3 referenceRight= Vector3.Cross(Vector3.up, lookDirection);
		if ( float.IsNaN(referenceRight.x) 
		    || float.IsNaN(referenceRight.y) 
		    || float.IsNaN(referenceRight.z) )
			return 0f;
			
		// Get the angle in degrees between 0 and 180
		float angle = Vector3.Angle(targetDirection, lookDirection);
		
		// Determine if the degree value should be negative. Here, a positive value
		// from the dot product means that our vector is on the right of the reference vector
		// whereas a negative value means we're on the left.
		float sign = Mathf.Sign(Vector3.Dot(targetDirection, referenceRight));
		
		float finalAngle = sign * angle;
		return finalAngle;
	}

	private void recordSensorData(ref NeuralNetwork net, ArrayList hits, float rotation){
		net.allInputs.AddRange(hits);

		if( (float)hits[2] == 1 && (float)hits[1] == 1 && (float)hits[0] == 0 ){
			net.allOutputs.Add( 0f );
			net.allOutputs.Add( 0f );
			net.allOutputs.Add( 1f );
		}
		else if( (float)hits[2] == 0 && (float)hits[1] == 1 && (float)hits[0] == 1 ){
			net.allOutputs.Add( 1f );
			net.allOutputs.Add( 0f );
			net.allOutputs.Add( 0f );
		}
		else if( (float)hits[2] == 1 && (float)hits[1] == 0 && (float)hits[0] == 0 ){
			net.allOutputs.Add( 0f );
			net.allOutputs.Add( 0f );
			net.allOutputs.Add( 1f );
		}
		else if( (float)hits[2] == 0 && (float)hits[1] == 0 && (float)hits[0] == 1 ){
			net.allOutputs.Add( 1f );
			net.allOutputs.Add( 0f );
			net.allOutputs.Add( 0f );
		}
		else{
			net.allOutputs.Add( 0f );
			net.allOutputs.Add( 0f );
			net.allOutputs.Add( 0f );
		}
	}

	private void recordRotation(ref NeuralNetwork net, float rotation, float angleDiff){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		//net.numInputs = 1 + hits.Count;
		//print (rotation);
		net.allOutputs.Add(rotation);

		net.allInputs.Add(angleDiff);
		//net.allInputs.AddRange(hits);
	}

	private void recordDirection(ref NeuralNetwork net, float verticalMovement,
	                             float deltaZ, float angleDiff){
		if (sensor == null)
			sensor = gameObject.transform.GetChild(0).gameObject.GetComponent<RadialGridSensor> ();

		//print (( 1 - (Mathf.Abs(angleDiff)/180) ) * verticalMovement);

		net.allOutputs.Add (/*( 1 - (angleDiff/180) ) **/ verticalMovement);
		net.allInputs.Add (deltaZ);
		//net.allInputs.Add (angleDiff);
	}

	private void trainNetworks(){
		float[,] sensorTrain = { {0f,0f,0f}, {0f,0f,1f}, {0f,1f,0f}, {0f,1f,1f},
								 {1f,0f,0f}, {1f,0f,1f}, {1f,1f,0f}, {1f,1f,1f}};

		for(int i = 0; i < 8; i++){
			ArrayList hits = new ArrayList();
			hits.Add(sensorTrain[i,0]);
			hits.Add(sensorTrain[i,1]);
			hits.Add(sensorTrain[i,2]);

			recordSensorData(ref nets[2], hits, 0f);
		}

		foreach (NeuralNetwork net in nets)
			net.LearningPhase(net.allInputs, net.allOutputs, net.allowedError);
		TrainingPhase = false;
	}

	private Vector3 GetNewDirection(NeuralNetwork net, float deltaZ, float angleDiff){
		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, new float[]{
									deltaZ
									/*angleDiff*/});
		LA.Matrix<float> outputs = net.ComputeOutputs(inputs);
		return new Vector3(0, 0, outputs[0,0]);
	}
	
	private float[] GetNewSensorData(NeuralNetwork net, ArrayList hits){
		float[] input_array = new float[net.numInputs];
		
		for(int i = 0; i < hits.Count; i++){
			float hit = (float)hits[i];
			input_array[i] = hit; 
		}

		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, input_array);
		
		LA.Matrix<float> outputs = net.ComputeOutputs(inputs);
		float[] output = new float[]{outputs[0,0], outputs[0,1], outputs[0,2]};
		return output;
	}

	private float GetNewRotation(NeuralNetwork net, float angleDiff){
		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		float[] input_array = new float[net.numInputs];
		input_array[0] = angleDiff;

		LA.Matrix<float> inputs = LA.Matrix<float>.Build.Dense (1, net.numInputs, input_array);

		LA.Matrix<float> outputs = net.ComputeOutputs(inputs);
		return outputs[0,0];
	}

	// Update is called once per frame
	void FixedUpdate () {
		float verticalMovement = Input.GetAxis ("Vertical");
		float horizontalMovement = Input.GetAxis ("Horizontal");
		float rotAmount = 0f;
		float sensorData = 0f;
		float deltaZ = 0f;
		Vector3 direction = new Vector3 (0, 0, verticalMovement);
		ArrayList hits = sensor.collisionCheck();
		Vector3 goalDirection;
		float angleDiff;

		if (target == null)
			target = GameObject.FindGameObjectWithTag("Goal");

		if( recordMovement ){
			if( WorldManager.currentLevel == 1 ){
				RaycastHit hitInfo;
				Ray r = new Ray(transform.position, transform.rotation * Vector3.forward*100);
				if(Physics.Raycast( r, out hitInfo, 100, 1<<LayerMask.NameToLayer("Goal") )){
					Debug.DrawRay(transform.position, transform.rotation * Vector3.forward*100, Color.red);
					sm.OnTriggerEnter(target.collider);
				}
				else
					Debug.DrawRay(transform.position, transform.rotation * Vector3.forward*100, Color.white);
			}
			Debug.DrawLine(transform.position, target.transform.position);
		}

		//deltaZ = Mathf.Abs(target.transform.position.z - transform.position.z);
		deltaZ = Vector3.Distance(target.transform.position, transform.position);
		rotAmount = horizontalMovement;
		goalDirection = target.transform.position - transform.position;
		angleDiff = vectorDiff(goalDirection, transform.forward);

		// Handle direction recording
		if ( WorldManager.currentLevel == 0 && recordMovement 
		    && !TrainingPhase && (direction.z != 0f || rotAmount != 0f) )
			recordDirection(ref nets[0], verticalMovement, deltaZ, Mathf.Abs(angleDiff) );

		// Handle sensor data recording
		//if ( recordMovement 
		//    && !TrainingPhase && (direction.z != 0f || rotAmount != 0f) )
		//	recordSensorData(ref nets[2], hits, rotAmount);

		// Handle rotation data recording
		if ( WorldManager.currentLevel == 1 &&  recordMovement 
		    && !TrainingPhase && (direction.z != 0f || rotAmount != 0f) )
			recordRotation(ref nets[1], rotAmount, angleDiff);

		// Handle Network training
		if ( !recordMovement && TrainingPhase )
			trainNetworks();

		// Controller
		if ( !debugMovement && !recordMovement && !TrainingPhase ){
			float[] rot_prob = GetNewSensorData(nets[2], hits);

			if ( (float)hits[1] != 1 
			    && ((Mathf.Abs(angleDiff) < 165 && Mathf.Abs (angleDiff) > 15f) || deltaZ > 15f )
			    && ( (float)hits[0] == 1 || (float)hits[2] == 1 ) )
				rotAmount = 0;
			else if ( (float)hits[1] == 1 && (float)hits[0] == 1 )
				rotAmount = -1;
			else if ( (float)hits[1] == 1 && (float)hits[2] == 1 )
				rotAmount = 1;
			else
				rotAmount = GetNewRotation(nets[1], angleDiff);

			//print ( rotAmount );
			if( (float)hits[1] == 1 ){
				sm.move(new Vector3(0f,0f,0f));
				direction.z = 0.4f;
			}
			else if( (float)hits[1] != 1 && ( (float)hits[0] == 1 || (float)hits[2] == 1 ) )
				direction.z = 0.5f;
			else if( (float)hits[1] != 1 && (float)hits[0] != 1 && (float)hits[2] != 1 && Mathf.Abs(angleDiff) > 10f )
				direction.z = 0.30f;
			else
				direction = GetNewDirection(nets[0], deltaZ, Mathf.Abs(angleDiff) );


			print ( new Vector3(rot_prob[0], rot_prob[1], rot_prob[2]) );
		}

		sm.rotate(rotAmount * Time.deltaTime * sm.rotateSpeed);
		sm.move(direction);
	}
}
