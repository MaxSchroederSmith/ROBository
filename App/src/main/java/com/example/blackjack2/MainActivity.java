package com.example.blackjack2;

import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.UUID;

public class MainActivity extends AppCompatActivity {

    private static final UUID MY_UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB"); // Replace with your own UUID

    private BluetoothSocket socket;
    private InputStream inputStream;
    private OutputStream outputStream;

    private TextView receivedMessagesTextView;
    private Button sendHitButton;
    private Button sendStandButton;

    private Button sendSplitButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        receivedMessagesTextView = findViewById(R.id.textView);
        sendHitButton = findViewById(R.id.Hit);
        sendStandButton = findViewById(R.id.Stand);
        sendSplitButton = findViewById(R.id.Split);


        // Set the OnClickListeners for the send buttons
        sendHitButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                sendMessage("Hit");
            }
        });

        sendStandButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                sendMessage("Stand");
            }
        });

        sendSplitButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                sendMessage("Split");
            }
        });

        // Start a background thread to establish the Bluetooth connection
        new Thread(new Runnable() {
            @SuppressLint("MissingPermission")
            @Override
            public void run() {
                try {
                    BluetoothAdapter bluetoothAdapter = BluetoothAdapter.getDefaultAdapter();
                    BluetoothDevice device = bluetoothAdapter.getRemoteDevice("00:11:22:33:44:55"); // Replace with the MAC address of your remote device
                    socket = device.createRfcommSocketToServiceRecord(MY_UUID);
                    socket.connect();
                    inputStream = socket.getInputStream();
                    outputStream = socket.getOutputStream();

                    // Send an initial message to the Python script
                    String initialMessage = "Hello from Android!";
                    sendMessage(initialMessage);

                    // Start a loop to read messages from the input stream and update the UI
                    while (true) {
                        byte[] receivedBuffer = new byte[1024];
                        int numBytes = inputStream.read(receivedBuffer);
                        String receivedMessage = new String(receivedBuffer, 0, numBytes);

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                receivedMessagesTextView.setText(receivedMessage);
                            }
                        });
                    }
                } catch (IOException e) {
                    Log.e("Bluetooth", "Error occurred while communicating with Python script", e);
                } finally {
                    try {
                        if (inputStream != null) {
                            inputStream.close();
                        }
                        if (outputStream != null) {
                            outputStream.close();
                        }
                        if (socket != null) {
                            socket.close();
                        }
                    } catch (IOException e) {
                        Log.e("Bluetooth", "Error occurred while closing streams and socket", e);
                    }
                }
            }
        }).start();
    }

    private void sendMessage(String message) {
        try {
            byte[] buffer = message.getBytes();
            outputStream.write(buffer);
        } catch (IOException e) {
            Log.e("Bluetooth", "Error occurred while sending message", e);
        }
    }
}
