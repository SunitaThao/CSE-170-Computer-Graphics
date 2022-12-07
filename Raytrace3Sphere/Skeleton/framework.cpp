#include "framework.h"

// Initialization
void onInitialization(int a);

// Window has become invalid: Redraw
void onDisplay();

void keyboard_func(unsigned char key, int x, int y);
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY);

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY);

// Move mouse with key pressed
void onMouseMotion(int pX, int pY);

// Mouse click event
void onMouse(int button, int state, int pX, int pY);

// Idle event indicating that some time elapsed: do animation here
void onIdle();

// Entry point of the application
int main(int argc, char * argv[]) {
	// Initialize GLUT, Glew and OpenGL 
	glutInit(&argc, argv);

	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(600, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	
	glewInit();
#endif

	// Initialize this program and create shaders
	onInitialization(1);

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	return 1;
}
