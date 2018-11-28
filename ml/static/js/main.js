function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

var csrftoken = getCookie('csrftoken');

$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});

var ws;

function connect_ws() {
    ws = new WebSocket('ws://localhost:8000');

    ws.onopen = () => {
        console.log('Hello');
    };
    
    ws.close = () => {
        console.log('GoodBye');
    };

    ws.onerror = () => {
        console.log('error');
    };

    ws.onmessage = () => {
        console.log('Message');
    };

}

function start_script(elem, event) {
    event.stopPropagation();
    $.ajax({
        type: 'POST',
        url: location.origin,
        data: {
            command: 'start',
        },
        success: r => {
            update_content();
        },
        error: (jqXHR, textStatus, errorThrown) => {},
        complete: () => {},
    });
}

function stop_script(elem, event) {
    $.ajax({
        type: 'POST',
        url: location.origin,
        data: {
            command: 'stop',
        },
        success: r => {
            update_content();
        },
        error: (jqXHR, textStatus, errorThrown) => {},
        complete: () => {},
    });
}

function update_content() {
    $.get(`${location.origin}`, data => {
        console.log(data)
        var elem = $('<div></div>').html( data ).find( '#content_container' );
        $('#content_container').html(elem.html());
    });
}

document.addEventListener('DOMContentLoaded', () => {
    connect_ws();
});
